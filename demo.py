with open("/media/xin/data1/test_data/bigobj/test_result/my1.11.49/update.txt") as f:
    list_line = f.readlines()
res = []
for i in range(len(list_line)-1):
    if "update: 1" in list_line[i] and "write image to" in list_line[i+1]:
        res.append(list_line[i])
        res.append(list_line[i+1])

with open("res.txt","w") as f:
    for line in res:
        f.write(line)
