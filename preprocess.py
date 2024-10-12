import json
import gzip
from tqdm import tqdm
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split
import re
import sys

class Preprocess():
    def __init__(self, dataset):
        self.dataset = dataset
    
    def parse(self, file_path):
        g = gzip.open(file_path, 'rb')
        for l in tqdm(g):
            yield json.loads(l)

    def count(self, topk):
        f = open(f'./data/{self.dataset}/meta_{self.dataset}.json', 'r')
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
        review_text = defaultdict(lambda: defaultdict(lambda: ''))
        file_path = f'./data/{self.dataset}/{self.dataset}.json.gz'
        empty_review = 0
        for l in self.parse(file_path):
            if l['asin'] not in item_with_meta:
                continue
            line += 1
            asin = l['asin'] # item id
            rev = l['reviewerID']
            # countU[rev] += 1
            countI[asin] += 1
            reviewText = ""
            summary = ""
            try:
                reviewText = l['reviewText']
            except:
                pass
            try:
                summary = l['summary']
            except:
                pass
            review_text[rev][asin] = reviewText + ' ' + summary
            if(review_text[rev][asin] == ' '):
                empty_review += 1
            review_text[rev][asin] = re.sub(r'[\n\t\s\"\']+', ' ', review_text[rev][asin])

        print(f'Empty review: {empty_review}')
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
        empty_review = 0
        user_id = {}
        item_id = {}
        item_meta = {}
        user_num = 0
        item_num = 0
        interactions = defaultdict(list)
        kept_reviews = defaultdict(lambda: defaultdict(lambda: ''))

        for l in self.parse(file_path):
            asin = l['asin']
            if asin not in frequent_item:
                continue
            rev = l['reviewerID']
            if countU[rev] > topk:
                if rev in user_id:
                    userid = user_id[rev]
                else:
                    userid = user_num
                    user_id[rev] = userid
                    interactions[userid] = []
                    user_num += 1
                if asin in item_id:
                    itemid = item_id[asin]
                else:
                    itemid = item_num
                    item_id[asin] = itemid
                    item_meta[itemid] = metadata[asin]   
                    item_num += 1 
                interactions[userid].append(itemid)
                kept_reviews[userid][itemid] = review_text[rev][asin]
                if review_text[rev][asin] == '':
                    empty_review += 1
        print(f'Empty review: {empty_review}')



        with open (f'./data/{self.dataset}/train.txt', 'w+') as trainf:
            with open (f'./data/{self.dataset}/test.txt', 'w+') as testf:
                for user in interactions.keys():
                    try:
                        train, test = train_test_split(interactions[user], test_size=0.2, shuffle=True)
                    except:
                        print(user)
                        print(interactions[user])
                    trainf.write(f'{user} ' + ' '.join(map(str, train)) + '\n')
                    testf.write(f'{user} ' + ' '.join(map(str, test)) + '\n')

        with open (f'./data/{self.dataset}/item_dict.txt', 'w+') as itemf:
            for key, value in item_meta.items():
                itemf.write(f'{key} {value}\n')

        with open (f'./data/{self.dataset}/review_dict.txt', 'w+') as reviewf:
            for key, value in kept_reviews.items():
                for k, v in value.items():
                    reviewf.write(f'{key} {k} {v}\n')

if __name__ == '__main__':
    dataset = str(sys.argv[1])
    preprocess = Preprocess(dataset)
    preprocess.count(10)
