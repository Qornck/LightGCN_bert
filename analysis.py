CountUser = {}
CountItem = {}

with open("./train.txt") as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n').split(' ')
            items = [int(i) for i in l[1:]]
            uid = int(l[0])
            CountUser[uid] = len(items)
            for i in items:
                if i in CountItem:
                    CountItem[i] += 1
                else:
                    CountItem[i] = 1
with open("./test.txt") as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n').split(' ')
            items = [int(i) for i in l[1:]]
            uid = int(l[0])
            CountUser[uid] += len(items)
            for i in items:
                if i in CountItem:
                    CountItem[i] += 1
                else:
                    CountItem[i] = 1


for i in CountUser.keys():
    if CountUser[i] < 5:
        print(i)