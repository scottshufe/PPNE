from datetime import datetime
import networkx as nx
import numpy as np
import argparse, pickle, time, os, collections, torch
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
parser.add_argument('--save',default='./tradeoff_cora_random', type=str)
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
adj_train = np.load('../cora_data/adj_train.npy')
test_edges = np.load('../cora_data/test_edges.npy')
test_edges_false = np.load('../cora_data/test_edges_false.npy')
labels = np.load('../cora_data/labels.npy')

g = nx.from_numpy_matrix(adj_train, parallel_edges=False, create_using=nx.Graph())
g.remove_edges_from(test_edges)
adj_train = nx.adj_matrix(g).todense()

adj_train = adj_train.astype('float64')

print('read data over')

print(adj_train.shape)
print(test_edges.shape)
print(test_edges_false.shape)
print(labels.shape)
#--------------------------------------------------------------------------------------------------
graph_first = nx.from_numpy_matrix(adj_train, parallel_edges=False, create_using=nx.Graph())
print(nx.info(graph_first))
print("now embedding method: ", args.method)

with open(os.path.join(args.save, 'cora_random_modify'), 'a') as file_record:
    file_record.write('random modify, now time: {}\n'.format(datetime.now()))
    file_record.write('-------------------------------------------\n')
    file_record.flush()


if args.method == 'deepwalk':
    print('start deepwalk!')
    X, Y = deepwalk.deepwalk(args, graph_first, verbose=args.verbose, random_seed=RANDOM_SEED)
elif args.method == 'line':
    X, Y = line.line(args, graph_first, IN_PATH, SAVE_PATH, SAVE_PATH_CT)

X_ori = X

roc_score, ap_score = util.evaluate(X, test_edges, test_edges_false)
f1_xgb = util.evaluate_scenario2(X, test_edges, test_edges_false, args.dim)
f1_score_mean, _, acc = util.evaluate_embedding_node_classification(X, labels)
nmi = util.evaluate_embedding_node_clustering(X_ori, X, labels)
similarity_score = util.intermediate_similarity_calculation(X, test_edges)

print("Roc score %f ap score %f"%(roc_score, ap_score))
print("Scenario 2: XGB F1: {:.4f}".format(f1_xgb))
print("Intermediate similarity score %f"%(similarity_score))
print("F1: {:.4f} {:.4f} Acc: {:.4f}".format(f1_score_mean[0], f1_score_mean[1], acc))
print("NMI: {:.4f}".format(nmi))

# with open(os.path.join(args.save, 'random_record_file_modify'), 'a') as file_record:
#     file_record.write('randome delete, now time: {}\n'.format(datetime.now()))
#     file_record.write("original result: roc score %8f ap score %8f macro f1 %8f micro f1 %8f \n"%
#                       (roc_score, ap_score, f1_score_mean[0], f1_score_mean[1]))
#     file_record.flush()

#-----------------------------------------------------------

victim_edges = np.concatenate((test_edges, test_edges_false), axis=0)

adj_train_first = np.copy(adj_train)

for _ in range(1):
    adds_or_deletes = np.random.choice([0, 1], 2000, p=[0.5, 0.5])

    num_deleted_edges = 0
    num_added_edges = 0

    for i in range(len(adds_or_deletes)):

        if i % 50 == 0:
            print("Randomly modified %f edges, num added %f, num deleted %f"%(i, num_added_edges, num_deleted_edges))
            print("Evaluation:")
            train_graph_new = nx.from_numpy_matrix(adj_train_first, parallel_edges=False, create_using=nx.Graph())
            if args.method == 'deepwalk':
                X, Y = deepwalk.deepwalk(args, train_graph_new, verbose=args.verbose, random_seed=RANDOM_SEED)
            elif args.method == 'line':
                X, Y = line.line(args, train_graph_new, IN_PATH, SAVE_PATH, SAVE_PATH_CT)

            roc_score, ap_score = util.evaluate(X, test_edges, test_edges_false)
            f1_xgb = util.evaluate_scenario2(X, test_edges, test_edges_false, args.dim)
            f1_score_mean, _, acc = util.evaluate_embedding_node_classification(X, labels)
            nmi_scores = []
            for n in range(20):
                nmi_score = util.evaluate_embedding_node_clustering(X_ori, X, labels, random_seed=n)
                nmi_scores.append(nmi_score)
            nmi = np.mean(nmi_scores)
            similarity_score = util.intermediate_similarity_calculation(X, test_edges)

            with open(os.path.join(args.save, 'cora_random_modify'), 'a') as file_record:
                file_record.write('ITER {} over, now time: {}\n'.format(i, datetime.now()))
                file_record.write("added edge num: {}\n".format(num_added_edges))
                file_record.write("deleted edge num: {}\n".format(num_deleted_edges))
                file_record.write('-------------------------------------------\n')
                file_record.write(
                    "Iteration %3d roc score %8f ap score %8f scenario2 f1 xgb %8f similarity score %f micro f1 %8f macro f1 %8f nmi score %8f\n"
                    % (i, roc_score, ap_score, f1_xgb, similarity_score, f1_score_mean[0], f1_score_mean[1], nmi))
                file_record.flush()

            print("Iteration %3d roc score %8f ap score %8f scenario2 f1 xgb %8f similarity score %f micro f1 %8f macro f1 %8f nmi score %8f\n"
                    % (i, roc_score, ap_score, f1_xgb, similarity_score, f1_score_mean[0], f1_score_mean[1], nmi))

        if adds_or_deletes[i] == 0:
            # random del
            print("ITER {} random delete an edge".format(i))
            mask_del = (adj_train_first==1)
            remove_list = []
            keep_edge = []
            for j in range(mask_del.shape[0]):
                if j in remove_list:
                    continue
                indexes = np.argwhere(adj_train_first[j]).T[0]
                if len(indexes) == 0:
                    continue
                rand = np.random.choice(len(indexes), 1, replace=False)

                remove_list.append(rand)
                remove_list.append(j)
                keep_edge.append((j, rand))

                mask_del[j][indexes[rand]] = False
                mask_del.T[j][indexes[rand]] = False

            mask_del_low = np.tril(mask_del, k=-1)

            adj_train_first = lib.random_delete_edge_with_mask(adj_train_first, 1, mask_del_low, False)
            num_deleted_edges += 1

        else:
            # random add
            print("ITER {} random add an edge".format(i))
            mask_add = (adj_train_first == 0)
            for edge in victim_edges:
                mask_add[edge[0]][edge[1]] = False
                mask_add[edge[1]][edge[0]] = False
            for i in range(adj_train.shape[0]):
                mask_add[i][i] = False

            mask_add_low = np.tril(mask_add, k=-1)
            adj_train_first = lib.random_add_delete_edge_with_mask(adj_train_first, 1, 0, mask_add_low, False)
            num_added_edges += 1

    print("Random Evaluation Done.")