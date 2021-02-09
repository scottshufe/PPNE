from datetime import datetime
from collections import Counter
import networkx as nx
import numpy as np
import scipy.sparse as sp
import argparse, pickle, time, os, collections, torch
import lib
import line, deepwalk
import util
import util_utility


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
parser.add_argument('--init', type=float, default=0.001)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--add-delete-prob', type=float, default=0.5)
parser.add_argument('--candidate-add-num', type=int, default=10000)
parser.add_argument('--save',default='./variant_results', type=str)
args = parser.parse_args()


if not os.path.exists(args.save):
    os.makedirs(args.save)
###########################################################################################################

# Set hyperparameter
RANDOM_SEED = args.seed

if not os.path.exists(os.path.join(args.save, 'tmp')):
    os.makedirs(os.path.join(args.save, 'tmp'))

IN_PATH = os.path.join(args.save, 'tmp', 'input.txt')
SAVE_PATH = os.path.join(args.save, 'tmp', 'output.txt')
SAVE_PATH_CT = os.path.join(args.save, 'tmp', 'context.txt')

with open(os.path.join(args.save, 'cora_utility_modify'), 'a') as file_record:
    file_record.write('utility_modify_iter, now time: {}\n'.format(datetime.now()))
    file_record.write('-------------------------------------------\n')
    file_record.flush()

#------------------------------------------------------------------------------------------------
# Make the split.
# modify
adj_train = np.load('../../data/cora_data/adj_train.npy')
test_edges = np.load('../../data/cora_data/test_edges.npy')
test_edges_false = np.load('../../data/cora_data/test_edges_false.npy')
labels = np.load('../../data/cora_data/labels.npy')

g = nx.from_numpy_matrix(adj_train, parallel_edges=False, create_using=nx.Graph())
g.remove_edges_from(test_edges)
adj_train = nx.adj_matrix(g).todense()

adj_train = adj_train.astype('float64')

#--------------------------------------------------------------------------------------------------
graph_first = nx.from_numpy_matrix(adj_train, parallel_edges=False, create_using=nx.Graph())
print(nx.info(graph_first))

print("now embedding method: ", args.method)

if args.method == 'deepwalk':
    print('start deepwalk!')
    X, Y = deepwalk.deepwalk(args, graph_first, verbose=args.verbose, random_seed=RANDOM_SEED)
elif args.method == 'line':
    X, Y = line.line(args, graph_first, IN_PATH, SAVE_PATH, SAVE_PATH_CT)

X_ori = X

roc_score, ap_score = util.evaluate(X, test_edges, test_edges_false)
scenario2_f1_xgb = util.evaluate_scenario2(X, test_edges, test_edges_false, args.dim)
similarity_score = util.intermediate_similarity_calculation(X, test_edges)
f1_score_mean, _, acc = util.evaluate_embedding_node_classification(X, labels)
nmi = util.evaluate_embedding_node_clustering(X_ori, X, labels)

print("Roc score %f ap score %f"%(roc_score, ap_score))
print("Scenario 2: XGB F1: {:.4f}".format(scenario2_f1_xgb))
print("Intermediate similarity score %f"%(similarity_score))
print("F1: {:.4f} {:.4f} Acc: {:.4f}".format(f1_score_mean[0], f1_score_mean[1], acc))
print("NMI: {:.4f}".format(nmi))
#----------------------------------------------------------------------------------------------------
if args.method == 'deepwalk':
    MF = lib.DeepWalkMF(T=args.window_size).cuda()
elif args.method == 'line':
    MF = lib.LINEMF().cuda()
#-----------------------------------------------------------
victim_edges = np.concatenate((test_edges, test_edges_false), axis=0)

adj_train_first = np.copy(adj_train)

with open(os.path.join(args.save, 'cora_utility_modify'), 'a') as file_record:
    file_record.write('start optimization, now time: {}\n'.format(datetime.now()))
    file_record.write('-------------------------------------------\n')
    file_record.flush()

for _ in range(1):
    #------------------------------------------------------------------------
    ITER = 1000
    adds_or_deletes = np.random.choice([0, 1], ITER, p=[1 - args.add_delete_prob, args.add_delete_prob])
    # adds_or_deletes = np.random.choice([0, 1], ITER, p=[0.5, 0.5])
    adj_train_opt = adj_train_first

    for i in range(50, 1050, 50):
        flips = adds_or_deletes[:i]
        delete_num = Counter(flips)[0]
        add_num = Counter(flips)[1]
        print("flip num: {}".format(i))
        print("deleted edge num: {}".format(delete_num))
        print("added edge num: {}".format(add_num))

        print("delete edges...")
        adj_train = np.copy(adj_train_opt)
        adj_train_csr = sp.csr_matrix(adj_train)
        del_candidate = util.generate_candidates_removal(adj_train_csr)
        top_del = util_utility.perturbation_top_flips(adj_train_csr, del_candidate, delete_num, args.dim, args.window_size)
        adj_train = util.flip_candidates(adj_train_csr, top_del).toarray()

        print("add edges...")
        adj_train_csr = sp.csr_matrix(adj_train)
        add_candidate = util.generate_candidates_addition(adj_train_csr, args.candidate_add_num)
        top_add = util_utility.perturbation_top_flips(adj_train_csr, add_candidate, add_num, args.dim, args.window_size)
        adj_train = util.flip_candidates(adj_train_csr, top_add).toarray()

        # -------------------------------------------------------------------------
        print("Begin Training")
        train_graph = nx.from_numpy_matrix(adj_train, parallel_edges=False, create_using=nx.Graph())
        if args.method == 'deepwalk':
            X, Y = deepwalk.deepwalk(args, train_graph, verbose=args.verbose, random_seed=RANDOM_SEED)
        elif args.method == 'line':
            X, Y = line.line(args, train_graph, IN_PATH, SAVE_PATH, SAVE_PATH_CT)

        roc_score, ap_score = util.evaluate(X, test_edges, test_edges_false)
        scenario2_f1_xgb = util.evaluate_scenario2(X, test_edges, test_edges_false, args.dim)
        similarity_score = util.intermediate_similarity_calculation(X, test_edges)
        f1_score_mean, _, acc = util.evaluate_embedding_node_classification(X, labels)
        nmi_scores = []
        for n in range(20):
            nmi_score = util.evaluate_embedding_node_clustering(X_ori, X, labels, random_seed=n)
            nmi_scores.append(nmi_score)
        nmi = np.mean(nmi_scores)
        # -----------------------------------------------------------------------------------------------------
        with open(os.path.join(args.save, 'cora_utility_modify'), 'a') as file_record:
            file_record.write('ITER {} over, now time: {}\n'.format(i, datetime.now()))
            file_record.write('-------------------------------------------\n')
            file_record.write(
                "Iteration %3d roc score %8f ap score %8f scenario2 f1 xgb %8f similarity score %f micro f1 %8f macro f1 %8f nmi score %8f adj sum %f\n"
                % (i, roc_score, ap_score, scenario2_f1_xgb, similarity_score, f1_score_mean[0], f1_score_mean[1], nmi,
                   np.sum(adj_train)))
            file_record.flush()

        print(
            "Iteration %3d roc score %8f ap score %8f scenario2 f1 xgb %8f similarity score %f micro f1 %8f macro f1 %8f nmi score %8f adj sum %f\n"
            % (
            i, roc_score, ap_score, scenario2_f1_xgb, similarity_score, f1_score_mean[0], f1_score_mean[1], nmi, np.sum(adj_train)))

    print("\n\n\nOptimization Done")
