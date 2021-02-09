from datetime import datetime
import heapq
import networkx as nx
import numpy as np
import scipy.sparse as sp
import argparse, pickle, time, os, collections, torch
import lib
import line, deepwalk
import util
import util_utility
from pdb import set_trace as st


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
parser.add_argument('--candidate-add-num', type=int, default=50000)
parser.add_argument('--save',default='./pubmed_ppne_flips_500', type=str)
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

with open(os.path.join(args.save, 'ppne_500flips_modify'), 'a') as file_record:
    file_record.write('flips 500, now time: {}\n'.format(datetime.now()))
    file_record.write('-------------------------------------------\n')
    file_record.flush()

with open(os.path.join(args.save, 'ppne_500flips_final'), 'a') as file_record_final:
    file_record_final.write('pubmed, deepwalk, 500flips, now time: {}\n'.format(datetime.now()))
    file_record_final.write('-------------------------------------------\n')
    file_record_final.flush()
#------------------------------------------------------------------------------------------------
# Make the split.
# modify
adj_train_csr = sp.load_npz('../pubmed_data/adj_train_csr.npz')
test_edges = np.load('../pubmed_data/test_edges.npy')
test_edges_false = np.load('../pubmed_data/test_edges_false.npy')
labels = np.load('../pubmed_data/labels.npy')

# g = nx.from_numpy_matrix(adj_train, parallel_edges=False, create_using=nx.Graph())
g = nx.from_scipy_sparse_matrix(adj_train_csr)
nodes = list(g.nodes())
g.remove_edges_from(test_edges)

adj_train = nx.adjacency_matrix(g, nodelist=nodes).todense()

adj_train = adj_train.astype('float32')

print('read data over')

print(adj_train_csr.shape)
print(test_edges.shape)
print(test_edges_false.shape)
print(labels.shape)
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
# if args.method == 'deepwalk':
#     MF = lib.DeepWalkMF_float(T=args.window_size).cuda()
# elif args.method == 'line':
#     MF = lib.LINEMF().cuda()
#-----------------------------------------------------------
victim_edges = np.concatenate((test_edges, test_edges_false), axis=0)

adj_train_first = np.copy(adj_train)

with open(os.path.join(args.save, 'ppne_500flips_modify'), 'a') as file_record:
    file_record.write('start optimization, now time: {}\n'.format(datetime.now()))
    file_record.write('-------------------------------------------\n')
    file_record.flush()

for _ in range(1):
    #------------------------------------------------------------------------
    ITER = 100
    adj_train_opt = adj_train_first
    deleted_edges = 0
    added_edges = 0
    for i in range(ITER):
        # each iter modify 100 entries
        adds_or_deletes = np.random.choice([0, 1], 500, p=[1 - args.add_delete_prob, args.add_delete_prob])
        add_num = np.sum(adds_or_deletes)
        del_num = 500 - add_num

        if args.method == 'deepwalk':
            MF = lib.DeepWalkMF_float(T=args.window_size).cuda()
        elif args.method == 'line':
            MF = lib.LINEMF().cuda()

        print("now ITER: {}".format(i))
        with open(os.path.join(args.save, 'ppne_500flips_modify'), 'a') as file_record:
            file_record.write('ITER {}, now time: {}\n'.format(i+1, datetime.now()))
            file_record.flush()

        print("deleted edge num: {}".format(deleted_edges))
        print("added edge num: {}".format(added_edges))

        # find add candidate entries
        adj_train_csr = sp.csr_matrix(adj_train_opt)
        candidates = util.generate_candidates_addition(adj_train_csr, n_candidates=args.candidate_add_num)
        mask_add = sp.csr_matrix(np.zeros(adj_train.shape))
        mask_add = util.flip_candidates(mask_add, candidates).toarray()
        mask_add = (mask_add == 1)

        mask_low_add = np.tril(mask_add, k=-1)
        nonzero_add = mask_low_add.nonzero()
        i_ts_add, j_ts_add = nonzero_add

        candidates_add = np.array([[i_ts_add[n], j_ts_add[n]] for n in range(len(i_ts_add))])
        loss_for_candidates_add = util_utility.perturbation_utility_loss(adj_train_csr, candidates_add,
                                                                     args.dim, args.window_size)

        y_ts_add = adj_train[i_ts_add, j_ts_add]

        # find del candidate entries
        adj_train_csr = sp.csr_matrix(adj_train_opt)
        candidates = util.generate_candidates_removal(adj_train_csr)
        mask_del = sp.csr_matrix(np.zeros(adj_train.shape))
        mask_del = util.flip_candidates(mask_del, candidates).toarray()
        mask_del = (mask_del == 1)

        mask_low_del = np.tril(mask_del, k=-1)
        nonzero_del = mask_low_del.nonzero()
        i_ts_del, j_ts_del = nonzero_del

        candidates_del = np.array([[i_ts_del[n], j_ts_del[n]] for n in range(len(i_ts_del))])
        loss_for_candidates_del = util_utility.perturbation_utility_loss(adj_train_csr, candidates_del,
                                                                     args.dim, args.window_size)
        adj_train = np.copy(adj_train_opt)

        y_ts_del = adj_train[i_ts_del, j_ts_del]

        adj_train = np.copy(adj_train_opt)
        adj_train[i_ts_add, j_ts_add] = args.init
        adj_train[j_ts_add, i_ts_add] = args.init

        # -------------------------------------------------------------------------
        print("Begin optimization")
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
        ###### Phase 1 ######
        print("Start Phase 1: calculate X_grad")
        X_torch, Y_torch = torch.tensor(X).float().cuda(), torch.tensor(Y).float().cuda()
        X_grad = util.get_grad_on_X(X_torch, test_edges, test_edges_false)
        # -----------------------------------------------------------------------------------------------------
        print("Start Phase 2: calculate grad_Z")
        d_rt = np.sum(adj_train, axis=1, dtype=np.float32)

        MF.set_d_rt(d_rt)

        adj_torch = torch.tensor(adj_train, dtype=torch.float32).cuda()

        Z_forward = MF(adj_torch)
        mask_on_Z = (Z_forward.cpu().data.numpy() != -float('inf'))

        grad_Z = util.build_grad_on_Z_adv_torch_float2(X_torch, Y_torch, X_grad, mask_on_Z)

        # clear memory
        Z_forward = Z_forward.cpu()
        X_torch = X_torch.cpu()
        Y_torch = Y_torch.cpu()
        X_grad = X_grad.cpu()
        grad_Z = grad_Z.cpu()
        torch.cuda.empty_cache()
        # ------------------------------------------------------------------------------------
        # st()
        print("Start Phase 3: calculate adj_grad")
        i1_ts, j1_ts = mask_on_Z.nonzero()

        loss = (grad_Z.float()[i1_ts, j1_ts] * Z_forward[i1_ts, j1_ts]).sum()

        loss.backward()
        adj_grad = adj_torch.grad.cpu().data.numpy()
        # ------------------------------------------------------------------------------------
        adj_grad = (adj_grad + adj_grad.T) / 2.
        y_ts_grad_add = adj_grad[i_ts_add, j_ts_add]
        y_ts_grad_del = adj_grad[i_ts_del, j_ts_del]

        # add - positive gradient
        pos_grad_indice = np.where(y_ts_grad_add >= 0)[0]
        pu_vals_add = abs(y_ts_grad_add[pos_grad_indice] / loss_for_candidates_add[pos_grad_indice])
        top_idx_add_ = heapq.nlargest(add_num, range(len(pu_vals_add)), pu_vals_add.take)
        top_idx_add = pos_grad_indice[top_idx_add_]

        i_top_add = i_ts_add[top_idx_add]
        j_top_add = j_ts_add[top_idx_add]

        # del - negative gradient
        neg_grad_indice = np.where(y_ts_grad_del <= 0)[0]
        pu_vals_del = abs(y_ts_grad_del[neg_grad_indice] / loss_for_candidates_del[neg_grad_indice])
        top_idx_del_ = heapq.nlargest(del_num, range(len(pu_vals_del)), pu_vals_del.take)
        top_idx_del = neg_grad_indice[top_idx_del_]

        i_top_del = i_ts_del[top_idx_del]
        j_top_del = j_ts_del[top_idx_del]

        # the optimal edge
        print("Add and del edges...")

        # add the optimal edge
        for k in range(len(i_top_add)):
            adj_train_opt[i_top_add[k]][j_top_add[k]] = 1
            adj_train_opt.T[i_top_add[k]][j_top_add[k]] = 1

        for k in range(len(i_top_del)):
            adj_train_opt[i_top_del[k]][j_top_del[k]] = 0
            adj_train_opt.T[i_top_del[k]][j_top_del[k]] = 0

        deleted_edges += del_num
        added_edges += add_num

        del MF
        torch.cuda.empty_cache()

        with open(os.path.join(args.save, 'ppne_500flips_modify'), 'a') as file_record:
            file_record.write('ITER {} over, now time: {}\n'.format(i + 1, datetime.now()))
            file_record.write('-------------------------------------------\n')
            file_record.write(
                "Iteration %3d roc score %8f ap score %8f similarity score %f s2 xgb f1 %8f micro f1 %8f macro f1 %8f nmi score %8f adj sum %f y_ts_max_add %8f, y_ts_min_add %f, y_grad_max_add %8f y_grad_min_add %8f y_ts_max_del %8f, y_ts_min_del %f, y_grad_max_del %8f y_grad_min_del %8f\n"
                % (i, roc_score, ap_score, similarity_score, scenario2_f1_xgb, f1_score_mean[0], f1_score_mean[1], nmi,
                   np.sum(adj_train_opt), np.max(y_ts_add),
                   np.min(y_ts_add), np.max(y_ts_grad_add), np.min(y_ts_grad_add), np.max(y_ts_del),
                   np.min(y_ts_del), np.max(y_ts_grad_del), np.min(y_ts_grad_del)))
            file_record.flush()

        print(
            "Iteration %3d roc score %8f ap score %8f similarity score %f s2 xgb f1 %8f micro f1 %8f macro f1 %8f nmi score %8f adj sum %f y_ts_max_add %8f, y_ts_min_add %f, y_grad_max_add %8f y_grad_min_add %8f y_ts_max_del %8f, y_ts_min_del %f, y_grad_max_del %8f y_grad_min_del %8f\n"
                % (i, roc_score, ap_score, similarity_score, scenario2_f1_xgb, f1_score_mean[0], f1_score_mean[1], nmi,
                   np.sum(adj_train_opt), np.max(y_ts_add),
                   np.min(y_ts_add), np.max(y_ts_grad_add), np.min(y_ts_grad_add), np.max(y_ts_del),
                   np.min(y_ts_del), np.max(y_ts_grad_del), np.min(y_ts_grad_del)))

    print("\n\n\nOptimization Done")


