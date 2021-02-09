from datetime import datetime
from collections import Counter
import networkx as nx
import numpy as np
import argparse, pickle, time, os, collections, torch
import random
import lib
import line, deepwalk
import util


parser = argparse.ArgumentParser("deepwalk", formatter_class=argparse.ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--matfile-variable-name', default='network', help='variable name of adjacency matrix inside a .mat file.')
parser.add_argument('--max-memory-data-size', default=1000000000, type=int, help='Size to start dumping walks to disk, instead of keeping them in memory.')
parser.add_argument('--num_walks', default=10, type=int, help='Number of random walks to start at each node')
parser.add_argument('--dim', default=128, type=int, help='Number of latent dimensions to learn for each node.')
parser.add_argument('--seed', default=0, type=int, help='Seed for random walk generator.')
parser.add_argument('--vertex-freq-degree', default=False, action='store_true')
parser.add_argument('--walk_length', default=40, type=int, help='Length of the random walk started at each node')
parser.add_argument('--window-size', default=10, type=int, help='Window size of skipgram model.')
parser.add_argument('--workers', default=1, type=int, help='Number of parallel processes.')
parser.add_argument('--iter', default=1, type=int, help='number of iterations in word2vec')
parser.add_argument('--dataset',default='cora',type=str)
parser.add_argument('--edge-range', type=int, nargs='+', default=[0,1])
parser.add_argument('--mask', type=str,default='adj')
parser.add_argument('--p', type=float, default=1,help='Return hyperparameter. Default is 1.')
parser.add_argument('--q', type=float, default=1,help='Inout hyperparameter. Default is 1.')
parser.add_argument('--directed', dest='directed', action='store_true')
parser.add_argument('--method',default='deepwalk',type=str)
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--opt-iter',type=int,default=20)
parser.add_argument('--ratio',type=float,default=0.05)
parser.add_argument('--save',default='./tradeoff_cora_DP', type=str)
args = parser.parse_args()

if not os.path.exists(args.save):
    os.makedirs(args.save)

# Set parameters
RANDOM_SEED = args.seed

if not os.path.exists(os.path.join(args.save, 'tmp')):
    os.makedirs(os.path.join(args.save, 'tmp'))

IN_PATH = os.path.join(args.save, 'tmp', 'input.txt')
SAVE_PATH = os.path.join(args.save, 'tmp', 'output.txt')
SAVE_PATH_CT = os.path.join(args.save, 'tmp', 'context.txt')

#------------------------------------------------------------------------------------------------
# Make the split.
adj_train = np.load('../cora_data/adj_train.npy')
test_edges = np.load('../cora_data/test_edges.npy')
test_edges_false = np.load('../cora_data/test_edges_false.npy')
labels = np.load('../cora_data/labels.npy')

g = nx.from_numpy_matrix(adj_train, parallel_edges=False, create_using=nx.Graph())
g.remove_edges_from(test_edges)
adj_train = nx.adj_matrix(g).todense()

print('read data over')

print(adj_train.shape)
print(test_edges.shape)
print(test_edges_false.shape)
print(labels.shape)
#--------------------------------------------------------------------------------------------------
adj_train = np.array(adj_train)
print(np.sum(adj_train))

with open(os.path.join(args.save, 'dp_record_file_modify'), 'a') as file_record:
    file_record.write('DP modify, now time: {}\n'.format(datetime.now()))
    file_record.write('-------------------------------------------\n')
    file_record.flush()

with open(os.path.join(args.save, 'dp_record_file_modify_final'), 'a') as file_record_final:
    file_record_final.write('DP modify, now time: {}\n'.format(datetime.now()))
    file_record_final.write('-------------------------------------------\n')
    file_record_final.flush()

# differential privacy param
epsilon_list = [0, 0.1, 0.5, 1, 10, 100]
for epsilon in epsilon_list:

    if epsilon == 0:
        adj_dp = np.copy(adj_train)
        adj_dp = np.matrix(adj_dp)
        adj_dp = adj_dp.astype('float64')

        graph_first = nx.from_numpy_matrix(adj_dp, parallel_edges=False, create_using=nx.DiGraph())
        print(nx.info(graph_first))
        print("now embedding method: ", args.method)

        if args.method == 'deepwalk':
            print('start deepwalk!')
            X, Y = deepwalk.deepwalk(args, graph_first, verbose=args.verbose, random_seed=RANDOM_SEED)
        elif args.method == 'line':
            print('start line')
            X, Y = line.line(args, graph_first, IN_PATH, SAVE_PATH, SAVE_PATH_CT)

        X_ori = X

        roc_score, ap_score = util.evaluate(X, test_edges, test_edges_false)
        f1_xgb = util.evaluate_scenario2(X, test_edges, test_edges_false, args.dim)
        f1_score_mean, _, acc = util.evaluate_embedding_node_classification(X, labels)
        nmi = util.evaluate_embedding_node_clustering(X_ori, X, labels)
        similarity_score = util.intermediate_similarity_calculation(X, test_edges)

        print("Roc score %f ap score %f" % (roc_score, ap_score))
        print("Scenario 2: XGB F1: {:.4f}".format(f1_xgb))
        print("Intermediate similarity score %f" % (similarity_score))
        print("F1: {:.4f} {:.4f} Acc: {:.4f}".format(f1_score_mean[0], f1_score_mean[1], acc))
        print("NMI: {:.4f}".format(nmi))

        with open(os.path.join(args.save, 'dp_record_file_modify'), 'a') as file_record:
            file_record.write('epsilon {}, now time: {}\n'.format(epsilon, datetime.now()))
            file_record.write("epsilon %f roc score %8f ap score %8f s2 xgb f1 %8f similarity score %f micro f1 %8f macro f1 %8f nmi score %8f\n"
            % (epsilon, roc_score, ap_score, f1_xgb, similarity_score, f1_score_mean[0], f1_score_mean[1], nmi))
            file_record.flush()

        with open(os.path.join(args.save, 'dp_record_file_modify_final'), 'a') as file_record_final:
            file_record_final.write('epsilon {}, now time: {}\n'.format(epsilon, datetime.now()))
            file_record_final.write("epsilon %f roc score %8f ap score %8f s2 xgb f1 %8f similarity score %f micro f1 %8f macro f1 %8f nmi score %8f\n"
            % (epsilon, roc_score, ap_score, f1_xgb, similarity_score, f1_score_mean[0], f1_score_mean[1], nmi))
            file_record_final.flush()

        print(
            "epsilon %f roc score %8f ap score %8f s2 xgb f1 %8f similarity score %f micro f1 %8f macro f1 %8f nmi score %8f\n"
            % (epsilon, roc_score, ap_score, f1_xgb, similarity_score, f1_score_mean[0], f1_score_mean[1], nmi))

    else:
        sims = []
        aps = []
        s2_f1s = []
        f1s = []
        nmis = []
        for k in range(20):
            while True:
                np.random.seed(k)
                adj_dp = np.copy(adj_train)
                dp_manipulation_vec = util.differential_privacy_manipulation(adj_train, epsilon=epsilon)

                print(Counter(dp_manipulation_vec))

                add_node_idx = np.where(dp_manipulation_vec > 0)[0]
                del_node_idx = np.where(dp_manipulation_vec < 0)[0]

                for idx in add_node_idx:
                    manip_num = int(dp_manipulation_vec[idx])
                    zero_adj = np.where(adj_dp[idx]==0)[0]
                    add_loc = np.random.choice(len(zero_adj), manip_num, replace=False)
                    adj_dp[idx][zero_adj[add_loc]] = 1

                for idx in del_node_idx:
                    manip_num = abs(int(dp_manipulation_vec[idx]))
                    nonzero_adj = np.where(adj_dp[idx]==1)[0]
                    del_loc = np.random.choice(len(nonzero_adj), manip_num, replace=False)
                    adj_dp[idx][nonzero_adj[del_loc]] = 0

                print(np.sum(adj_dp))

                adj_dp = np.matrix(adj_dp)
                adj_dp = adj_dp.astype('float64')

                graph_first = nx.from_numpy_matrix(adj_dp, parallel_edges=False, create_using=nx.DiGraph())
                print(nx.info(graph_first))
                print("now embedding method: ", args.method)
                try:
                    if args.method == 'deepwalk':
                        print('start deepwalk!')
                        X, Y = deepwalk.deepwalk(args, graph_first, verbose=args.verbose, random_seed=RANDOM_SEED)
                    elif args.method == 'line':
                        print('start line')
                        X, Y = line.line(args, graph_first, IN_PATH, SAVE_PATH, SAVE_PATH_CT)
                except:
                    print("run embedding error. re-generate network")
                    continue

                roc_score, ap_score = util.evaluate(X, test_edges, test_edges_false)
                f1_xgb = util.evaluate_scenario2(X, test_edges, test_edges_false, args.dim)
                f1_score_mean, _, acc = util.evaluate_embedding_node_classification(X, labels)
                nmi_scores = []
                for n in range(20):
                    nmi_score = util.evaluate_embedding_node_clustering(X_ori, X, labels, random_seed=n)
                    nmi_scores.append(nmi_score)
                nmi = np.mean(nmi_scores)
                similarity_score = util.intermediate_similarity_calculation(X, test_edges)

                sims.append(similarity_score)
                aps.append(ap_score)
                s2_f1s.append(f1_xgb)
                f1s.append(f1_score_mean[0])
                nmis.append(nmi)

                print("Roc score %f ap score %f"%(roc_score, ap_score))
                print("Scenario 2: XGB F1: {:.4f}".format(f1_xgb))
                print("Intermediate similarity score %f"%(similarity_score))
                print("F1: {:.4f} {:.4f} Acc: {:.4f}".format(f1_score_mean[0], f1_score_mean[1], acc))
                print("NMI: {:.4f}".format(nmi))

                with open(os.path.join(args.save, 'dp_record_file_modify'), 'a') as file_record:
                    file_record.write('epsilon {}, now time: {}\n'.format(epsilon, datetime.now()))
                    file_record.write("epsilon %f roc score %8f ap score %8f s2 xgb f1 %8f similarity score %f micro f1 %8f macro f1 %8f nmi score %8f\n"
                    % (epsilon, roc_score, ap_score, f1_xgb, similarity_score, f1_score_mean[0], f1_score_mean[1], nmi))
                    file_record.flush()

                print(
                    "epsilon %f roc score %8f ap score %8f s2 xgb f1 %8f similarity score %f micro f1 %8f macro f1 %8f nmi score %8f\n"
                    % (epsilon, roc_score, ap_score, f1_xgb, similarity_score, f1_score_mean[0], f1_score_mean[1], nmi))

                break

        sim = np.mean(sims)
        ap = np.mean(aps)
        s2_f1 = np.mean(s2_f1s)
        f1 = np.mean(f1s)
        nmi = np.mean(nmis)

        with open(os.path.join(args.save, 'dp_record_file_modify_final'), 'a') as file_record_final:
            file_record_final.write('epsilon {}, now time: {}\n'.format(epsilon, datetime.now()))
            file_record_final.write(
                "epsilon %f similarity score %f ap score %8f s2 xgb f1 %8f micro f1 %8f nmi score %8f\n"
                % (epsilon, sim, ap, s2_f1, f1, nmi))
            file_record_final.flush()
                #-----------------------------------------------------------
print("Random Evaluation Done.")