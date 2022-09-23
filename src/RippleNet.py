import time
import numpy as np
import torch.nn as nn
import torch as t
from src.evaluate import get_hit, get_ndcg
from src.load_base import load_data, get_records


class RippleNet(nn.Module):

    def __init__(self, dim, n_entities, H, n_rel, l1, l2):
        super(RippleNet, self).__init__()

        self.dim = dim
        self.H = H
        self.l1 = l1
        self.l2 = l2
        ent_emb = t.randn(n_entities, dim)
        rel_emb = t.randn(n_rel, dim, dim)
        nn.init.xavier_uniform_(ent_emb)
        nn.init.xavier_uniform_(rel_emb)
        self.ent_emb = nn.Parameter(ent_emb)
        self.rel_emb = nn.Parameter(rel_emb)
        self.criterion = nn.BCELoss(reduction='sum')

    def forward(self, pairs, ripple_sets):

        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]
        item_embeddings = self.ent_emb[items]
        heads_list, relations_list, tails_list = self.get_head_relation_and_tail(users, ripple_sets)
        user_represents = self.get_vector(items, heads_list, relations_list, tails_list)

        predicts = t.sigmoid((user_represents * item_embeddings).sum(dim=1))

        return predicts

    def get_head_relation_and_tail(self, users, ripple_sets):

        heads_list = []
        relations_list = []
        tails_list = []
        for h in range(self.H):
            l_head_list = []
            l_relation_list = []
            l_tail_list = []

            for user in users:

                l_head_list.extend(ripple_sets[user][h][0])
                l_relation_list.extend(ripple_sets[user][h][1])
                l_tail_list.extend(ripple_sets[user][h][2])

            heads_list.append(l_head_list)
            relations_list.append(l_relation_list)
            tails_list.append(l_tail_list)

        return heads_list, relations_list, tails_list

    def get_vector(self, items, heads_list, relations_list, tails_list):

        o_list = []
        item_embeddings = self.ent_emb[items].view(-1, self.dim, 1)
        for h in range(self.H):
            head_embeddings = self.ent_emb[heads_list[h]].view(len(items), -1, self.dim, 1)
            relation_embeddings = self.rel_emb[relations_list[h]].view(len(items), -1, self.dim, self.dim)
            tail_embeddings = self.ent_emb[tails_list[h]].view(len(items), -1, self.dim)

            Rh = t.matmul(relation_embeddings, head_embeddings).view(len(items), -1, self.dim)
            hRv = t.matmul(Rh, item_embeddings)
            pi = t.softmax(hRv, dim=1)
            o_embeddings = (pi * tail_embeddings).sum(dim=1)
            o_list.append(o_embeddings)

        return sum(o_list)

    def computer_loss(self, labels, predicts, users, ripple_sets):

        base_loss = self.criterion(predicts, labels)
        kg_loss = 0
        for h in range(self.H):
            h_head_list = []
            h_relation_list = []
            h_tail_list = []
            for user in users:
                h_head_list.extend(ripple_sets[user][h][0])
                h_relation_list.extend(ripple_sets[user][h][1])
                h_tail_list.extend(ripple_sets[user][h][2])

            head_emb = self.ent_emb[h_head_list].view(-1, 1, self.dim)  # (n, dim)-->(n, 1, dim)
            rel_emb = self.rel_emb[h_relation_list].view(-1, self.dim, self.dim)  # (n, dim, dim)
            tail_emb = self.ent_emb[h_relation_list].view(-1, self.dim, 1)  # (n, dim)-->(n, dim, 1)

            Rt = t.matmul(rel_emb, tail_emb)  # (n, dim, 1)
            hRt = t.matmul(head_emb, Rt)  # (n, 1, 1)

            kg_loss = kg_loss - t.sigmoid(hRt).mean()

        return base_loss + self.l1 * kg_loss


def eval_topk(model, records, ripple_sets, topk):
    HR, NDCG = [], []
    model.eval()
    for user in records:

        items = list(records[user])
        pairs = [[user, item] for item in items]
        predict = model.forward(pairs, ripple_sets).cpu().view(-1).detach().numpy().tolist()
        # print(predict)
        n = len(pairs)
        user_scores = {items[i]: predict[i] for i in range(n)}
        item_list = list(dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True)).keys())[: topk]

        HR.append(get_hit(items[-1], item_list))
        NDCG.append(get_ndcg(items[-1], item_list))

    model.train()
    # print(np.mean(HR), np.mean(NDCG))
    return np.mean(HR), np.mean(NDCG)


def get_ripple_set(train_dict, kg_dict, H, size):

    ripple_set_dict = {user: [] for user in train_dict}

    for u in (train_dict):

        next_e_list = train_dict[u]

        for h in range(H):
            h_head_list = []
            h_relation_list = []
            h_tail_list = []
            for head in next_e_list:
                if head not in kg_dict:
                    continue
                for rt in kg_dict[head]:
                    relation = rt[0]
                    tail = rt[1]
                    h_head_list.append(head)
                    h_relation_list.append(relation)
                    h_tail_list.append(tail)

            if len(h_head_list) == 0:
                h_head_list = ripple_set_dict[u][-1][0]
                h_relation_list = ripple_set_dict[u][-1][1]
                h_tail_list = ripple_set_dict[u][-1][0]
            else:
                replace = len(h_head_list) < size
                indices = np.random.choice(len(h_head_list), size, replace=replace)
                h_head_list = [h_head_list[i] for i in indices]
                h_relation_list = [h_relation_list[i] for i in indices]
                h_tail_list = [h_tail_list[i] for i in indices]

            ripple_set_dict[u].append((h_head_list, h_relation_list, h_tail_list))

            next_e_list = ripple_set_dict[u][-1][2]

    return ripple_set_dict


def train(args):
    np.random.seed(123)

    data = load_data(args)
    n_entity, n_user, n_item, n_relation = data[0], data[1], data[2], data[3]
    train_set, eval_records, test_records = data[4], data[5], data[6]
    kg_dict = data[7]

    train_records = get_records(train_set)
    ripple_sets = get_ripple_set(train_records, kg_dict, args.H, args.K_h)

    model = RippleNet(args.dim, n_entity, args.H, n_relation, args.l1, args.l2)
    if t.cuda.is_available():
        model = model.to(args.device)

    optimizer = t.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    print(args.dataset + '-----------------------------------------')

    print('dim: %d' % args.dim, end='\t')
    print('H: %d' % args.H, end='\t')
    print('K_h: %d' % args.K_h, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)

    eval_HR_list = []
    eval_NDCG_list = []
    test_HR_list = []
    test_NDCG_list = []

    for epoch in (range(args.epochs)):

        start = time.clock()
        loss_sum = 0
        np.random.shuffle(train_set)
        for i in range(0, len(train_set), args.batch_size):

            if (i + args.batch_size + 1) > len(train_set):
                batch_uvls = train_set[i:]
            else:
                batch_uvls = train_set[i: i + args.batch_size]

            pairs = [[uvl[0], uvl[1]] for uvl in batch_uvls]
            labels = t.tensor([int(uvl[2]) for uvl in batch_uvls]).view(-1).float()
            if t.cuda.is_available():
                labels = labels.to(args.device)

            users = [pair[0] for pair in pairs]
            predicts = model(pairs, ripple_sets)

            loss = model.computer_loss(labels, predicts, users, ripple_sets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        eval_HR, eval_NDCG = eval_topk(model, eval_records, ripple_sets, args.topk)
        test_HR, test_NDCG = eval_topk(model, test_records, ripple_sets, args.topk)
        eval_HR_list.append(eval_HR)
        eval_NDCG_list.append(eval_NDCG)
        test_HR_list.append(test_HR)
        test_NDCG_list.append(test_NDCG)
        end = time.clock()

        print('epoch: %d \t eval: HR %.4f NDCG %.4f \t test: HR %.4f NDCG %.4f \t loss: %d, \t time: %d'
              % ((epoch + 1), eval_HR, eval_NDCG, test_HR, test_NDCG, loss_sum, (end - start)))

    n_epoch = eval_HR_list.index(max(eval_HR_list))
    print('epoch: %d \t eval: HR %.4f NDCG %.4f \t test: HR %.4f NDCG %.4f' % (
        n_epoch + 1, eval_HR_list[n_epoch], eval_NDCG_list[n_epoch], test_HR_list[n_epoch], test_NDCG_list[n_epoch]))
    return eval_HR_list[n_epoch], eval_NDCG_list[n_epoch], test_HR_list[n_epoch], test_NDCG_list[n_epoch]
