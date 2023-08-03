import os

path='C:/Users/USER/Desktop/TR/val/labels'


file_list = os.listdir(path)

test=''
for idx in range(len(file_list)):
    path_merge=path+"/2 "+"("+str(idx+1)+")"+".txt"
    # print(path_merge)
    # path_merge = 'C:/Users/USER/Desktop/train/labels' + "/idx.txt"

    with open(path_merge,'r') as f:
        temp_str = ''
        lines=f.readlines()

        for i in lines:
            test=''
            if i[:4] == '1300':
                test = i.replace('1300','0')
            elif i[:4] == '1301':
                test = i.replace('1301','1')
            elif i[:4] == '1302':
                test = i.replace('1302','2')
            elif i[:4] == '1303':
                test = i.replace('1303','3')
            elif i[:4] == '1305':
                test = i.replace('1305','4')
            elif i[:4] == '1306':
                test = i.replace('1306','5')
            elif i[:4] == '1400':
                test = i.replace('1400','6')
            elif i[:4] == '1401':
                test = i.replace('1401','7')
            elif i[:4] == '1402':
                test = i.replace('1402','8')
            elif i[:4] == '1403':
                test = i.replace('1403','9')
            elif i[:4] == '1404':
                test = i.replace('1404','10')
            elif i[:4] == '1405':
                test = i.replace('1405','11')
            elif i[:4] == '1406':
                test = i.replace('1406','12')
            elif i[:4] == '1407':
                test = i.replace('1407','13')
            elif i[:4] == '1408':
                test = i.replace('1408','14')
            elif i[:4] == '1409':
                test = i.replace('1409','15')
            elif i[:4] == '1501':
                test = i.replace('1501','16')
            elif i[:4] == '1502':
                test = i.replace('1502','17')
            elif i[:4] == '1700':
                test = i.replace('1700','18')
            elif i[:4] == '1701':
                test = i.replace('1701','19')
            else :
                test = i
            temp_str+= test
        print(temp_str)

    with open(path_merge,'w') as f:
        f.write(temp_str)
