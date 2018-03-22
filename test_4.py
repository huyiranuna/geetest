import numpy as np
import json
import datetime
from sklearn.externals import joblib

def get_file_data(file_path):
    file_data = []
    file = open(file_path)
    for line in file.readlines():
        d = json.loads(line)
        file_data.append(d)
    return file_data

def get_txt_float(file_path):
    file_data = []
    file = open(file_path)
    for line in file.readlines():
        l = [float(i) for i in line.strip("\n").split(' ')]
        file_data.append(l)
    file_data = np.array(file_data)

    return file_data


def get_txt_str(file_path):
    file = open(file_path)
    for line in file.readlines():
        file_data = [i.strip("\'") for i in line.strip("\n").strip("{").strip("}").split(", ")]
    # file_data = file_data.tolist()

    return file_data


def data_divide(data):
    divide = np.histogram(data, bins=4)
    divide_point = divide[1][1:4]
    return divide_point

def get_train_divide(data):
    divide = np.zeros(shape=(16,3))
    for i in range(16):
        divide[i] = data_divide(data[:, i])
    return divide

def data_hash(divide_point, data):
    if data < divide_point[0]:
        hash_code = '00'
    elif data >= divide_point[0] and data < divide_point[1]:
        hash_code = '01'
    elif data >= divide_point[1] and data < divide_point[2]:
        hash_code = '10'
    else:
        hash_code = '11'
    return hash_code

def get_hash_code(divide, data):
    hash_code = []
    for i in range(16):
        code = data_hash(divide[i], data[i])
        hash_code.append(code)
    a = ''
    hash_code = a.join(hash_code)
    return hash_code

def get_hash_list(divide, data):
    hash_list = []
    rows = data.shape[0]
    for i in range(rows):
        hash_code = get_hash_code(divide, data[i])
        hash_list.append(hash_code)
    return hash_list


def judge_hash(robot_hash_set, data_hash):
        if data_hash in robot_hash_set:
            return 1
        else:
            return 0


def hash_predict(data, divide0, divide1, robot_final_0p1r, robot_final_1p0r):
    search2 = joblib.load("GBDT1.model")
    GBDT_ypred = search2.predict(data).tolist()
    hash_ypred = []
    for i in range(data.shape[0]):
        hash_code0 = get_hash_code(divide0, data[i])
        j0 = judge_hash(robot_final_0p1r, hash_code0)
        hash_code1 = get_hash_code(divide1, data[i])
        j1 = judge_hash(robot_final_1p0r, hash_code1)

        if j0 == j1:
            hash_ypred.append(j0)
        else:
            hash_ypred.append(GBDT_ypred[i])
    return hash_ypred


def Kmeans_predict(search, Xs_test):
    y_pred = search.predict(Xs_test).tolist()
    return y_pred

if __name__ == '__main__':
    starttime = datetime.datetime.now()
    print(starttime)
    # train_people = np.array(get_file_data('D:\\hyr\\geetest\\gt-kaggle\\train_data\\train_people\\feature_355808\\part-00000'))
    # train_robot = np.array(get_file_data('D:\\hyr\\geetest\\gt-kaggle\\train_data\\train_robot\\feature_93900\\part-00000'))
    test_people = np.array(get_file_data('D:\\hyr\\geetest\\gt-kaggle\\test_data\\test_people\\feature_43955\\part-00000'))
    test_robot_c = np.array(get_file_data('D:\\hyr\\geetest\\gt-kaggle\\test_data\\test_robot_c\\feature_37702\\part-00000'))
    test_robot_Cl = np.array(get_file_data('D:\\hyr\\geetest\\gt-kaggle\\test_data\\test_robot_Cl\\feature_17139\\part-00000'))

    divide_00 = get_txt_float('D:\\PycharmProjects\\exam_hyr\\divide__00.txt')
    divide_10 = get_txt_float('D:\\PycharmProjects\\exam_hyr\\divide__10.txt')

    robot_0p1r = set(get_txt_str('D:\\PycharmProjects\\exam_hyr\\robot_0p1r.txt'))
    robot_1p0r = set(get_txt_str('D:\\PycharmProjects\\exam_hyr\\robot_1p0r.txt'))

    print('-------------------------------------------------------------')
    print('test_robot_c:')
    hash_pred1 = hash_predict(test_robot_c, divide_00, divide_10, robot_0p1r, robot_1p0r)
    print(len(hash_pred1))
    print(hash_pred1.count(0))
    print(hash_pred1.count(1))
    print('-------------------------------------------------------------')
    print('test_robot_Cl:')
    hash_pred2 = hash_predict(test_robot_Cl, divide_00, divide_10, robot_0p1r, robot_1p0r)
    print(len(hash_pred2))
    print(hash_pred2.count(0))
    print(hash_pred2.count(1))
    print('-------------------------------------------------------------')
    print('test_people:')
    hash_pred3 = hash_predict(test_people, divide_00, divide_10,  robot_0p1r, robot_1p0r)
    print(len(hash_pred3))
    print(hash_pred3.count(0))
    print(hash_pred3.count(1))

    endtime = datetime.datetime.now()
    print(endtime)



