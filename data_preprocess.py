import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# 列名
cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
        'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'target']

# 标签以及标签所对应的类别 其中正常数据为 normal 网络攻击共有四大类(39小类)： dos, u2r, r2l, probe
attacks_types = {
    'normal': 'normal',
    'back': 'dos',
    'buffer_overflow': 'u2r',
    'ftp_write': 'r2l',
    'guess_passwd': 'r2l',
    'imap': 'r2l',
    'ipsweep': 'probe',
    'land': 'dos',
    'loadmodule': 'u2r',
    'multihop': 'r2l',
    'neptune': 'dos',
    'nmap': 'probe',
    'perl': 'u2r',
    'phf': 'r2l',
    'pod': 'dos',
    'portsweep': 'probe',
    'rootkit': 'u2r',
    'satan': 'probe',
    'smurf': 'dos',
    'spy': 'r2l',
    'teardrop': 'dos',
    'warezclient': 'r2l',
    'warezmaster': 'r2l',
}

path = 'data/kddcup.data_10_percent.gz'
df = pd.read_csv(path, names=cols)

# Adding Attack Type column
df['attack_types'] = df.target.apply(lambda r: attacks_types[r[:-1]])


# 数据可视化
def bar_graph(feature):
    df[feature].value_counts().plot(kind="bar")
    # 保存图片
    plt.savefig(f'picture/{feature}.jpg')


bar_graph('protocol_type')

plt.figure(figsize=(15, 6))
bar_graph('service')

bar_graph('flag')

# 找到除 target 和 Attack Type 以外的非数值的标签
num_cols = df._get_numeric_data().columns

cate_cols = list(set(df.columns) - set(num_cols))
cate_cols.remove('target')
cate_cols.remove('attack_types')
print(cate_cols)
"""
['flag', 'protocol_type', 'service']
"""

# 将非数值标签编码
print(df['protocol_type'].value_counts())
pmap = {
    'icmp': 0,
    'tcp': 1,
    'udp': 2,
}
df['protocol_type'] = df['protocol_type'].map(pmap)

# 将非数值标签编码
print(df['flag'].value_counts())
fmap = {
    'SF': 0,
    'S0': 1,
    'REJ': 2,
    'RSTR': 3,
    'RSTO': 4,
    'SH': 5,
    'S1': 6,
    'S2': 7,
    'RSTOS0': 8,
    'S3': 9,
    'OTH': 10,
}
df['flag'] = df['flag'].map(fmap)

# 将非数值标签编码
print(df['attack_types'].value_counts())
amap = {
    'normal': 1,
    'dos': 2,
    'probe': 3,
    'u2r': 4,
    'r2l': 5,
}
df['attack_types'] = df['attack_types'].map(amap)

# service 种类分布过于广泛 直接舍弃
df.drop('service', axis=1, inplace=True)

# 去 NA
df = df.dropna('columns')

# 去特殊值
df = df[[col for col in df if df[col].nunique() > 1]]

# 计算相关度并绘图
corr = df.corr()
figure = plt.figure(figsize=(15, 12))
sns.heatmap(corr)
plt.show()
# 保存图片
figure.savefig(f'picture/correlation.jpg')


# 计算 l1 和 l2 的相关度并输出
def cal_cor(l1, l2):
    print(l1, l2, df[l1].corr(df[l2]))


cal_cor('num_root', 'num_compromised')
cal_cor('srv_serror_rate', 'serror_rate')
cal_cor('srv_rerror_rate', 'rerror_rate')
cal_cor('dst_host_srv_serror_rate', 'srv_serror_rate')
cal_cor('dst_host_serror_rate', 'rerror_rate')
cal_cor('dst_host_rerror_rate', 'srv_rerror_rate')
cal_cor('dst_host_srv_rerror_rate', 'rerror_rate')
cal_cor('dst_host_same_srv_rate', 'dst_host_srv_count')

# 与 num_compromised 接近线性相关 Correlation = 0.9938277978750971
df.drop('num_root', axis=1, inplace=True)
# 与 serror_rate 接近线性相关 Correlation = 0.9983615072725553
df.drop('srv_serror_rate', axis=1, inplace=True)
# 与 rerror_rate 接近线性相关 Correlation = 0.9947309539823285
df.drop('srv_rerror_rate', axis=1, inplace=True)
# 与 srv_serror_rate 接近线性相关 Correlation = 0.9993041091845912
df.drop('dst_host_srv_serror_rate', axis=1, inplace=True)
# 与 rerror_rate 接近线性相关 Correlation = 0.9436670688873966
df.drop('dst_host_serror_rate', axis=1, inplace=True)
# 与 srv_rerror_rate 接近线性相关 Correlation = 0.9821663427309738
df.drop('dst_host_rerror_rate', axis=1, inplace=True)
# 与 rerror_rate 接近线性相关 Correlation = 0.9851995540753726
df.drop('dst_host_srv_rerror_rate', axis=1, inplace=True)
# 与 dst_host_srv_count 接近线性相关 Correlation = 0.973685457296524
df.drop('dst_host_same_srv_rate', axis=1, inplace=True)

df.drop('target', axis=1, inplace=True)

# 划分训练集测试集
train_dataset, test_dataset = train_test_split(df, test_size=0.25)
# 保存
df.to_csv("data/df.csv", index=False)
train_dataset.to_csv("data/train_dataset.csv", index=False)
test_dataset.to_csv("data/test_dataset.csv", index=False)
