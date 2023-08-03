import os

path='C:/Users/USER/Desktop/TR/val/labels'

file_list = os.listdir(path)

for idx in range(len(file_list)):
    temp_str_list = []
    path_merge=path+"/2 "+"("+str(idx+1)+")"+".txt"
    temp_str=''
    with open(path_merge,'r') as f:
        lines=f.readlines()
        for i in lines:
            temp_str2 = ''
            lines2=i.split('\t')

            temp1 = lines2[4].replace("\n", "")
            lines2.insert(0,temp1)
            lines2.pop()

            lines2[1] = str(round(float(lines2[1]) * float(1/2048),8))
            lines2[2] = str(round(float(lines2[2]) * float(1/1536),8))
            lines2[3] = str(round(float(lines2[3]) * float(1/2048),8))
            lines2[4] = str(round(float(lines2[4]) * float(1/1536),8))


            for i in lines2:
                temp_str+=i + " "
            temp_str += "\n"
            temp_str_list.append(temp_str)
            temp_str = ''

            for j in temp_str_list:
                temp_str2 += j

            print("temp_str_list", temp_str_list)
            print("temp_str2", temp_str2)

            with open(path_merge, 'w') as f:
                f.write(temp_str2)
