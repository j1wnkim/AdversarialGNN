import os
import sys
import traceback
import uuid
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import random
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset
from torch_geometric.utils import degree, to_dense_adj, add_self_loops
from torch_geometric.transforms import Compose
from datasets import load_dataset, create_subdatasets
from models import NodeClassifier, SurrogateModel
from trainer import Trainer
from transforms import FeatureTransform, FeaturePerturbation, LabelPerturbation, FeaturePerturbationAttack 
from utils import print_args, WandbLogger, add_parameters_as_argument, \
    measure_runtime, from_args, str2bool, Enum, EnumAction, colored_text, bootstrap
from attacks import AttackMode, prepare_data, train_model, test_model, predict, average_feature_difference
from sklearn.metrics.pairwise import cosine_similarity
from shadow_attack import generateData, Train_and_Evaluate
from sklearn.neural_network import MLPClassifier
from sklearn import metrics



class LogMode(Enum):
    INDIVIDUAL = 'individual'
    COLLECTIVE = 'collective'

    
# class AttackMode(Enum):
#     ADDNODES = 'addNodes'
#     FLIPNODES = 'flipNodes'
#     INFERENCE = 'inference'

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def confidence_interval(data, func=np.mean, size=1000, ci=95, seed=12345):
    bs_replicates = bootstrap(data, func=func, n_boot=size, seed=seed)
    p = 50 - ci / 2, 50 + ci / 2
    bounds = np.nanpercentile(bs_replicates, p)
    return (bounds[1] - bounds[0]) / 2

def feature_privacy_budget(x_eps):
    return x_eps


@measure_runtime
def run(args):
    dataset = from_args(load_dataset, args)

    test_acc = []
    run_metrics = {}
    run_id = str(uuid.uuid1())

    logger = None
    if args.log and args.log_mode == LogMode.COLLECTIVE:
        logger = WandbLogger(project=args.project_name, config=args, enabled=args.log, reinit=False, group=run_id)
    
    attacker = None
    if args.attack:
        attacker = args.attack_type
    
    
    if attacker == AttackMode.POISON:
        data = dataset.clone()
        poisoned_data = dataset.clone()
        ### maybe write a for loop to which we modify the args and do something like:
        ### for x_eps in args.x_eps_list:
            ### for y_eps in args.y_eps_list: 
        poisoned_data = Compose([
            from_args(FeatureTransform, args),
            from_args(FeaturePerturbationAttack, args),
            from_args(LabelPerturbation, args)
        ])(poisoned_data)
        
        n, d = poisoned_data.x.size()

        
        x_eps = from_args(feature_privacy_budget, args)
        len_features = d ## the feature dimension basically. 
        
        print(f"The number of features in each node is {d}\n")        
        print(f"The privacy budget is {x_eps}\n")
        print(f"The privacy budget per feature is {float(x_eps)/d}\n")
#       print(n,d)
        
        
        correct_preds = 0
        total_preds = 0
        original_nums = 0
        attacked_nodes = 0
        
        for i in range(n):
            flag = False
            for j in range(d):
                if data.x[i][j] > 0.0:
                    original_nums += 1
                if poisoned_data.x[i][j] > 0.5: ## from the rectifier 
                    if data.x[i][j] > 0.0:
                        flag = True
                        correct_preds += 1
                    total_preds += 1
            if flag is True:
                attacked_nodes += 1
        accu = correct_preds / total_preds
        leaked = attacked_nodes / n
        print(f"\nThe number of vulnerable is {original_nums}\n")
        print(f"\nThe accuracy of the poison attack is {accu*100}%\n")
        print(f"\nThe degree of the poison attack is {leaked*100}%\n")
        
        model = SurrogateModel(input_dim=len_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005) ## maybe change it as args.lr 
        ### Training and Evaluating the model performance 
        print("---------------------------------------------------------------")
        # use poisoned data 
        model = from_args(NodeClassifier, args, input_dim=poisoned_data.num_features, num_classes=poisoned_data.num_classes)

        trainer = from_args(Trainer, args, logger=logger if args.log_mode == LogMode.INDIVIDUAL else None, attacker=attacker)
        best_metrics = trainer.fit(model, poisoned_data) ## currently train the model, maybe also train w/o the poisoned data??? 
            # process results
        for metric, value in best_metrics.items():
                run_metrics[metric] = run_metrics.get(metric, []) + [value]
        test_acc.append(best_metrics['test/acc'])
        output_dir = None 
        if not args.log: ## make sure that it is default 
            output_dir = args.output_dir + "Poison"
            os.makedirs(output_dir, exist_ok=True)
            df_results = pd.DataFrame(test_acc, columns=['test/acc']).rename_axis('version').reset_index()
            df_results['Name'] = "Poison "+ args.dataset + " "+ str(run_id)
            run_id2 = "Poison " + args.dataset + " "+ args.model + str(run_id)
            for arg_name, arg_val in vars(args).items():
                df_results[arg_name] = [arg_val] * len(test_acc)
            df_results.to_csv(os.path.join(output_dir, f'{run_id2}.csv'), index=False)
        
        return
        
    elif attacker == AttackMode.INFERENCE:
        data = None
        surrogate_data = None
        len_features = 0
        for i in range(10):
            temp_data = Compose([
                from_args(FeatureTransform, args),
                from_args(FeaturePerturbationAttack, args)
#                 from_args(LabelPerturbation, args)
            ])(dataset)
#             print(temp_data.x[0])
            if i == 0:
                data, surrogate_data = create_subdatasets(temp_data, dataset)
                len_features = len(surrogate_data.features[0])
            else:
                _, temp_surr_data = create_subdatasets(temp_data, dataset)
                surrogate_data = ConcatDataset([surrogate_data, temp_surr_data])
            
#         print(data)
#         print(surrogate_data)
#         print(len(surrogate_data[0]))
        
#         erty = input("WAIT")
        
        model = SurrogateModel(input_dim=len_features)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        
        epochs = 10
        
        train_loader, test_loader = prepare_data(surrogate_data)
        
        train_model(model, train_loader, optimizer, epochs)
        
        test_model(model, test_loader)
        
        noisy_data = Compose([
            from_args(FeatureTransform, args),
            from_args(FeaturePerturbationAttack, args)
#             from_args(LabelPerturbation, args)
        ])(data)
        
        raw_data = data
        
        input_range = raw_data.x.min().item(), raw_data.x.max().item()
        
        print(input_range)
        
        input_range = noisy_data.x.min().item(), noisy_data.x.max().item()
        
        print(input_range)
        print("--------------------------------------------------------")
        
        edge_index, _ = add_self_loops(noisy_data.edge_index, num_nodes=noisy_data.num_nodes)
    
        dense_adj = to_dense_adj(edge_index)[0]
        
        similarity = 0
        diffs = 0
        total = 0
        
        for node_idx in range(noisy_data.num_nodes):
            neighbors = dense_adj[node_idx].nonzero(as_tuple=True)[0]
            neighbor_features = noisy_data.x[neighbors]
            original_neighbors = raw_data.x[neighbors]
            
#             for neighbor in neighbor_features:
#                 print(neighbor)
#             print("---------------------------")
#             for neighbor in original_neighbors:
#                 print(neighbor)
            
#             foo = input("ENTER TO CONTINUE\n")
                
            if neighbor_features.shape[0] == 1:
                continue
            
            variance_vector = torch.var(neighbor_features, dim=0)
            
            feature = torch.cat( ( variance_vector, torch.tensor( [len(neighbors)], dtype=torch.float32 ) ) )
            
            pred = predict(model, feature)
            
            if pred > 0.75:
                mean_vector = torch.mean(neighbor_features, dim=0)
                original_vector = raw_data.x[node_idx]
                
                mean_original_vector = torch.mean(raw_data.x[neighbors], dim=0)
                
                cosine_sim = cosine_similarity(
                    mean_vector.unsqueeze(0),
                    original_vector.unsqueeze(0)
                )
                similarity += cosine_sim
                total += 1
                
                diff = average_feature_difference(mean_vector, original_vector)
                diffs += diff
                if len(neighbors) > 15:
                    for idx in range(len(original_vector)):
#                         if original_vector[idx] > 0.5:
#                             print(f'Mean value with a true feature is {mean_vector[idx]}')
#                         else:
#                             print(f'Mean value with a false feature is {mean_vector[idx]}')
                        print(f'The original value: {original_vector[idx]} \tThe mean value: {mean_vector[idx]}\n')

                    foo = input("ENTER TO CONTINUE\n")
#                     print(f'The diff with g.t. 30 neighbors: {diff}\n')
#                     print(mean_vector.unsqueeze(0))
#                     print(original_vector.unsqueeze(0))
#                     print("-----------------------------------------------")
                
        mean_sim = similarity / float(total)
        mean_diff = diffs / float(total)
        
        print(f'\nThe cosine similarity between predicted features and raw features using a confidence predictor is {mean_sim}\n')
        print(f'\nThe Mean feature difference between predicted features and raw features using a confidence predictor is {mean_diff}\n')
        
        
        return
    elif attacker == AttackMode.SHADOW:
        data = Compose([from_args(FeatureTransform, args)])(dataset.clone()) # we're only going to transform the features.
        original_data, excluded_original_data = generateData(data.clone(), 0.8, 0.25, 0.2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = from_args(NodeClassifier, args, input_dim = data.num_features, num_classes = data.num_classes) # create target model 
        path_1 = "original_model"
        path_2 = "shadow_model"
        
        Train_and_Evaluate(model, original_data, 500, 50, torch.optim.Adam(model.parameters()), device, path_1)
        
        
        shadow_model = from_args(NodeClassifier, args, input_dim = data.num_features, num_classes = data.num_classes) # shadow model to replicate. 
        shadow_data, hold_out_shadowData = generateData(data.clone(), graph_sample = 0.6, train_split = 0.7, val_split = 0.2)
        Train_and_Evaluate(shadow_model, shadow_data, 500, 50, torch.optim.Adam(model.parameters()), device, path_2)
        
        shadow_model.eval()

        shadow_train = shadow_model(shadow_data)[shadow_data.train_mask] # get the train 
        shadow_test = shadow_model(hold_out_shadowData)

        y_shadow_train = [1] * shadow_train.shape[0]
        y_shadow_test = [0] * shadow_test.shape[0]

        y_train_attack = y_shadow_train + y_shadow_test

        shadow_train = shadow_train.detach().cpu().numpy() ## put it to cpu device. 
        shadow_test = shadow_test.detach().cpu().numpy()

        x_train_attack = np.concatenate((shadow_train,shadow_test)) 
        original_data.to(device)
        excluded_original_data.to(device)

        target_train = shadow_model(original_data)[original_data.train_mask]
        target_test = shadow_model(excluded_original_data)

        y_target_train=[1]*target_train.shape[0]
        y_target_test=[0]*target_test.shape[0]

        target_train_copy = target_train.detach().cpu().numpy()
        target_test_copy = target_test.detach().cpu().numpy() 

        y_test_attack = y_target_train+y_target_test
        x_test_attack = np.concatenate((target_train_copy, target_test_copy)) 
        
        clf = MLPClassifier(random_state=1, solver='adam', max_iter=300).fit(x_train_attack, y_train_attack) 
        clf.fit(x_train_attack, y_train_attack)
        
        os.remove(path_1)
        os.remove(path_2) # delete after using it. 
        
        y_score = clf.predict(x_test_attack) 
        print(metrics.classification_report(y_test_attack, y_score, labels=range(2))) 
        print(metrics.roc_auc_score(y_test_attack, y_score)) 
    
    
    progbar = tqdm(range(args.repeats), file=sys.stdout)
    for version in progbar: ### This is the place for default training and evaluating our GNN 
        if args.log and args.log_mode == LogMode.INDIVIDUAL:
            args.version = version
            logger = WandbLogger(project=args.project_name, config=args, enabled=args.log, group=run_id)

        try:
            data = dataset.clone().to(args.device)
            # Added below 
#             print("\n\nEdge index is ", data.edge_index) 
#             degrees = degree(data.edge_index[0]) 
#             sorted_ids, _ = torch.sort(degrees, descending=True) 
#             poisoned_nodes = sorted_ids[:num_poisoned] 
#             data.y[poisoned_nodes] = 1 - data.y[poisoned_nodes] 
            # Added above 
            # preprocess data 
            data = Compose([
                from_args(FeatureTransform, args),
                from_args(FeaturePerturbation, args), ## default x_eps is infinite, thus normal data is retained. 
                from_args(LabelPerturbation, args)
            ])(data)
            # define model
            model = from_args(NodeClassifier, args, input_dim=data.num_features, num_classes=data.num_classes)
            # train the model
            trainer = from_args(Trainer, args, logger=logger if args.log_mode == LogMode.INDIVIDUAL else None, attacker=attacker)
            best_metrics = trainer.fit(model, data)

            # process results
            for metric, value in best_metrics.items():
                run_metrics[metric] = run_metrics.get(metric, []) + [value]

            test_acc.append(best_metrics['test/acc'])
            progbar.set_postfix({'last_test_acc': test_acc[-1], 'avg_test_acc': np.mean(test_acc)})

        except Exception as e:
            error = ''.join(traceback.format_exception(Exception, e, e.__traceback__))
            logger.log_summary({'error': error})
            raise e
        finally:
            if args.log and args.log_mode == LogMode.INDIVIDUAL:
                logger.finish()

    if args.log and args.log_mode == LogMode.COLLECTIVE:
        summary = {}
        for metric, values in run_metrics.items():
            summary[metric + '_mean'] = np.mean(values)
            summary[metric + '_ci'] = confidence_interval(values, size=1000, ci=95, seed=args.seed)

        logger.log_summary(summary)

    if not args.log: ## make sure that it is default 
        output_dir = None 
        if (args.x_eps >0 or args.y_eps >0) and (args.x_eps != np.inf and args.y_eps != np.inf):
            output_dir = args.output_dir + "LDP"
        else:
            output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        df_results = pd.DataFrame(test_acc, columns=['test/acc']).rename_axis('version').reset_index()
        df_results['Name'] = "Baseline " + args.dataset + " " + run_id
        for arg_name, arg_val in vars(args).items():
            df_results[arg_name] = [arg_val] * len(test_acc)
        if (args.x_eps >0 or args.y_eps >0) and (args.x_eps != np.inf and args.y_eps != np.inf):
            run_id2 = "LDP " + args.dataset + " " + args.model +  run_id
        else:
            run_id2 = "Baseline " + args.dataset + " " + args.model +  run_id
        print(run_id2)
        df_results.to_csv(os.path.join(output_dir, f'{run_id2}.csv'), index=False)


def main():
#     os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')

    # dataset args
    group_dataset = init_parser.add_argument_group('dataset arguments')
    add_parameters_as_argument(load_dataset, group_dataset) ## inspects the class and then automates adding in arguments
                                                            ## and their defaults. 

    # data transformation args
    group_perturb = init_parser.add_argument_group(f'data transformation arguments')
    add_parameters_as_argument(FeatureTransform, group_perturb)
    add_parameters_as_argument(FeaturePerturbation, group_perturb)
    add_parameters_as_argument(FeaturePerturbationAttack, group_perturb) 
    add_parameters_as_argument(LabelPerturbation, group_perturb)

    # model args
    group_model = init_parser.add_argument_group(f'model arguments')
    add_parameters_as_argument(NodeClassifier, group_model)

    # trainer arguments (depends on perturbation)
    group_trainer = init_parser.add_argument_group(f'trainer arguments')
    add_parameters_as_argument(Trainer, group_trainer)
    group_trainer.add_argument('--device', help='desired device for training', choices=['cpu', 'cuda'], default='cuda')

    # experiment args
    group_expr = init_parser.add_argument_group('experiment arguments')
    group_expr.add_argument('-s', '--seed', type=int, default=None, help='initial random seed') #type = the data type the command line arg should be converted. 
    group_expr.add_argument('-r', '--repeats', type=int, default=1, help="number of times the experiment is repeated")
    group_expr.add_argument('-o', '--output-dir', type=str, default='./output', help="directory to store the results")
    group_expr.add_argument('--log', type=str2bool, nargs='?', const=True, default=False, help='enable wandb logging')
    group_expr.add_argument('--log-mode', type=LogMode, action=EnumAction, default=LogMode.INDIVIDUAL,
                            help='wandb logging mode')
    group_expr.add_argument('--project-name', type=str, default='LPGNN', help='wandb project name')
    
    # attack args
    group_attack = init_parser.add_argument_group('attack arguments')
    group_attack.add_argument('--attack', type=str2bool, nargs='?', const=True, default=False, help='enable attack')
    group_attack.add_argument('--attack-type', type=AttackMode, action=EnumAction, default=AttackMode.ADDNODES,
                            help='set attacking mode')
    
    
    ### Try to manipulate the parser where we add the argument on the ex, ey, kx, ky. Them being list.

    parser = ArgumentParser(parents=[init_parser], formatter_class=ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    print_args(args)
    args.cmd = ' '.join(sys.argv)  # store calling command

    if args.seed:
        seed_everything(args.seed)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print(colored_text('CUDA is not available, falling back to CPU', color='red'))
        args.device = 'cpu'

    try:
        run(args)
    except KeyboardInterrupt:
        print('Graceful Shutdown...')


if __name__ == '__main__':
    main()
