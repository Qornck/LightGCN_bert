import json
import gzip
from tqdm import tqdm
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split

class Preprocess():
    def __init__(self, dataset):
        self.dataset = dataset
    
    def parse(self, file_path):
        g = gzip.open(file_path, 'rb')
        for l in tqdm(g):
            yield json.loads(l)

    def count(self, topk):
        f = open(f'../A-LLMREC/data/amazon/meta_{self.dataset}.json', 'r')
        json_data = f.readlines()
        f.close()
        metadata = {}
        item_with_meta = set()
        data_list = [json.loads(line[:-1]) for line in json_data]
        for l in data_list:
            asin = l['asin']
            if len(l['description']) == 0 and len(l['title']) == 0:
                continue
            elif len(l['description']) == 0:
                metadata[asin] = l['title']
                metadata[asin] = metadata[asin].replace('\n', ' ')
            else:
                title = l['title']
                description = l['description'][0]
                metadata[asin] = title + ' ' + description
                metadata[asin] = metadata[asin].replace('\n', ' ')
            item_with_meta.add(asin)

        # first count: item
        countU = defaultdict(lambda: 0)
        countI = defaultdict(lambda: 0)
        line = 0
        frequent_item = []
        file_path = f'../A-LLMREC/data/amazon/{self.dataset}.json.gz'
        for l in self.parse(file_path):
            if l['asin'] not in item_with_meta:
                continue
            line += 1
            asin = l['asin'] # item id
            # rev = l['reviewerID']
            # countU[rev] += 1
            countI[asin] += 1

        for key, value in countI.items():
            if value > topk:
                frequent_item.append(key)
            
        # second count: user
        for l in self.parse(file_path):
            asin = l['asin']
            if asin not in frequent_item:
                continue
            rev = l['reviewerID']
            countU[rev] += 1
        
        # third: record
        user_id = {}
        item_id = {}
        item_meta = {}
        user_num = 0
        item_num = 0
        interactions = defaultdict(list)
        for l in self.parse(file_path):
            asin = l['asin']
            if asin not in frequent_item:
                continue
            rev = l['reviewerID']
            flag = 0
            if countU[rev] > topk:
                if rev in user_id:
                    userid = user_id[rev]
                else:
                    user_num += 1
                    userid = user_num
                    user_id[rev] = userid
                    interactions[userid] = []
                if asin in item_id:
                    itemid = item_id[asin]
                else:
                    item_num += 1
                    itemid = item_num
                    item_id[asin] = itemid
                    try:
                        item_meta[itemid] = metadata[asin]
                    except:
                        flag = 1
                if flag == 0:        
                    interactions[userid].append(itemid)

        with open ('./data/Movie_and_TV/train.txt', 'w+') as trainf:
            with open ('./data/Movie_and_TV/test.txt', 'w+') as testf:
                for user in interactions.keys():
                    try:
                        train, test = train_test_split(interactions[user], test_size=0.2, shuffle=False)
                    except:
                        print(user)
                        print(interactions[user])
                    trainf.write(f'{user} ' + ' '.join(map(str, train)) + '\n')
                    testf.write(f'{user} ' + ' '.join(map(str, test)) + '\n')

        with open ('./data/Movie_and_TV/item_dict.txt', 'w+') as itemf:
            for key, value in item_meta.items():
                itemf.write(f'{key} {value}\n')
        

    
    # def preprocess(self, topk = 20):
    #     countU, countI, itemname_dict = self.count()
    #     usermap = {}
    #     usernum = 0
    #     itemmap = {}
    #     itemnum = 0
    #     Interactions = defaultdict(list)
    #     item_dict = {}

    #     file_path = f'../A-LLMREC/data/amazon/{self.dataset}.json.gz'

    #     for l in self.parse(file_path):
    #         asin = l['asin']
    #         rev = l['reviewerID']
    #         time = l['unixReviewTime']
    #         if countU[rev] < topk or countI[asin] < topk or asin not in itemname_dict.keys():
    #             continue
    #         if rev in usermap:
    #             userid = usermap[rev]
    #         else:
    #             usernum += 1
    #             userid = usernum
    #             usermap[rev] = userid
    #             Interactions[userid] = []
    #         if asin in itemmap:
    #             itemid = itemmap[asin]
    #         else:
    #             itemnum += 1
    #             itemid = itemnum
    #             itemmap[asin] = itemid
    #             item_dict[itemid] = itemname_dict[asin]

    #         Interactions[userid].append(itemid)

    #     # for userid in Interactions.keys():
    #     #     Interactions[userid].sort(key=lambda x: x[0])

    #     with open ('./train.txt', 'w+') as trainf:
    #         with open ('./test.txt', 'w+') as testf:
    #             for user in Interactions.keys():
    #                 try:
    #                     train, test = train_test_split(Interactions[user], test_size=0.2, shuffle=False)
    #                 except:
    #                     print(user)
    #                     print(Interactions[user])
    #                 trainf.write(f'{user} ' + ' '.join(map(str, train)) + '\n')
    #                 testf.write(f'{user} ' + ' '.join(map(str, test)) + '\n')

    #     with open ('./item_dict.txt', 'w+') as itemf:
    #         for key, value in item_dict.items():
    #             itemf.write(f'{key} {value}\n')
    #     return Interactions, item_dict

    
    # # def preprocess(self, topk=20):
    # #     countU = defaultdict(lambda: 0)
    # #     countP = defaultdict(lambda: 0)
    # #     # reviews = defaultdict(lambda: defaultdict(lambda: ''))
    # #     line = 0

    # #     file_path = f'../A-LLMREC/data/amazon/{self.dataset}.json.gz'
        
    # #     # counting interactions for each user and item
    # #     for l in self.parse(file_path):
    # #         line += 1
    # #         asin = l['asin']
    # #         rev = l['reviewerID']
    # #         time = l['unixReviewTime']
    # #         countU[rev] += 1
    # #         countP[asin] += 1
        
    # #     usermap = dict()
    # #     usernum = 0
    # #     itemmap = dict()
    # #     itemnum = 0
    # #     User = dict()
    # #     review_dict = {}
    # #     name_dict = {}
        
    # #     f = open(f'../A-LLMREC/data/amazon/meta_{self.dataset}.json', 'r')
    # #     json_data = f.readlines()
    # #     f.close()
    # #     data_list = [json.loads(line[:-1]) for line in json_data]
    # #     meta_dict = {}
    # #     for l in data_list:
    # #         meta_dict[l['asin']] = l
        
    # #     for l in self.parse(file_path):
    # #         line += 1
    # #         asin = l['asin']
    # #         rev = l['reviewerID']
    # #         time = l['unixReviewTime']

    # #         # remove cold start users and items    
    # #         if countU[rev] < topk or countP[asin] < topk:
    # #             continue
            
    # #         if rev in usermap:
    # #             userid = usermap[rev]
    # #         else:
    # #             usernum += 1
    # #             userid = usernum
    # #             usermap[rev] = userid
    # #             User[userid] = []
            
    # #         if asin in itemmap:
    # #             itemid = itemmap[asin]
    # #         else:
    # #             itemnum += 1
    # #             itemid = itemnum
    # #             itemmap[asin] = itemid
    # #         User[userid].append([time, itemid])
            
            
    # #         if itemmap[asin] in review_dict:
    # #             try:
    # #                 review_dict[itemmap[asin]]['review'][usermap[rev]] = l['reviewText']
    # #             except:
    # #                 pass
    # #             try:
    # #                 review_dict[itemmap[asin]]['summary'][usermap[rev]] = l['summary']
    # #             except:
    # #                 pass
    # #         else:
    # #             review_dict[itemmap[asin]] = {'review': {}, 'summary':{}}
    # #             try:
    # #                 review_dict[itemmap[asin]]['review'][usermap[rev]] = l['reviewText']
    # #             except:
    # #                 pass
    # #             try:
    # #                 review_dict[itemmap[asin]]['summary'][usermap[rev]] = l['summary']
    # #             except:
    # #                 a = 0
    # #         try:
    # #             if len(meta_dict[asin]['description']) == 0:
    # #                 name_dict['description'][itemmap[asin]] = 'Empty description'
    # #             else:
    # #                 name_dict['description'][itemmap[asin]] = meta_dict[asin]['description'][0]
    # #             name_dict['title'][itemmap[asin]] = meta_dict[asin]['title']
    # #         except:
    # #             pass
        
    # #     # with open(f'../../data/amazon/{self.dataset}_text_name_dict.txt', 'wb') as tf:
    # #     #     pickle.dump(name_dict, tf)
        
    # #     for userid in User.keys():
    # #         User[userid].sort(key=lambda x: x[0])

    # #     # 移除review为空的数据
    # #     reviews = []
    # #     interactions = defaultdict(list)
    # #     for user in User.keys():
    # #         for i in User[user]:
    # #             # flag = 0
    # #             # try:
    # #             #     reviews.append(review_dict[i[1]]['review'][user])
    # #             # except:
    # #             #     try:
    # #             #         reviews.append(review_dict[i[1]]['summary'][user])
    # #             #     except:
    # #             #         flag = 1
    # #             # if flag == 0:
    # #                 interactions[user].append(i[1])
    #     # with open ('./train.txt', 'a') as trainf:
    #     #     with open ('./test.txt', 'a') as testf:
    #     #         for user in interactions.keys():
    #     #             train, test = train_test_split(interactions[user], test_size=0.2)
    #     #             trainf.write(f'{user} ' + ' '.join(map(str, train)) + '\n')
    #     #             testf.write(f'{user} ' + ' '.join(map(str, test)) + '\n')
        
    # #     train.close()
    # #     test.close()

    # #     '''
    # #     user: user_item interactions
    # #     review_dict: item: user review text
    # #     name_dict: item: item name
    # #     '''

if __name__ == '__main__':
    dataset = 'Movies_and_TV'
    preprocess = Preprocess(dataset)
    preprocess.count(10)
