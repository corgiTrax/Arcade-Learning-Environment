SUM = 0
count = 0

with open('/scratch/cluster/zharucs/dataset_gaze/validate.txt','r') as f:
    for line in f:
        line = line.strip()
        if len(line) < 30:
            continue

        a=line.split(' ')
        try:
            SUM += float(a[-5])
        except IndexError:
            print a
        count += 1

print count
print "Average is: %f" % (SUM/count)
