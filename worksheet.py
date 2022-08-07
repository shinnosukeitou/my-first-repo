#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns

# データのダウンロード
df = pd.read_csv('student_data.csv')
df.head()


# In[6]:


# 各カラムのデータ確認
df.describe()


# In[8]:


# テーブル情報の取得
df.info()


# # 欠損値があるか確認
# df.isna().sum()
# 
# # どの列にも欠損値がないことが分かった
各カラムの情報

school - 生徒の通ってる学校(GP, MS)
sex - 生徒の性別(F:女性, M:男性)
age - 生徒の年齢(15~22歳
address　- 生徒の住んでる地域(U:都会, R:田舎)
famsize - 家族の大きさ(LE3:3人以下, GT3:3人以上)
Pstatus - 親の同居状態(T:同居, A:別居)
Medu - 母の教育(0:無し, 1:初等教育, 2:5~9年生, 3:中等教育, 4:高等教育)
Fedu - 父の教育(0:無し, 1:初等教育, 2:5~9年生, 3:中等教育, 4:高等教育)
Mjob - 母の仕事(teacher:教師, health:医療関連, at_home:アットホーム, other:その他, services’:行政サービス)
Fjob - 父の仕事(teacher:教師, health:医療関連, at_home:アットホーム, other:その他, services:行政サービス)
reason - 学校を選ぶ理由(home:家に近い, reputation, :学校の評判, course:コースの好み, other:その他)
guardian - 生徒の保護者(mother：母, father:父)
traveltime　-　通学時間(1:15分以下, 2:15~30分, 3:30~60分, 4:1時間以上)
studytime - 毎週の学習時間(1:2時間未満, 2:2~5時間, 3:5~10時間, 3:10時間以上)
failures - 過去のクラスの失敗数(n:1~3この場合,　4:それ以外)
schoolsup - 追加の教育サポート(yes, no)
famsup - 家族の教育サポート(yes, no)
paid - 追加有料クラス(yes, no)
activities - 課外活動(yed, no)
nursery - 保育園に通っていた(yes, no)
higher - 高等教育を受けていた(yes, no)
internet - 家でインターネットを使えるか(yes, no)
romantic - 恋人がいる(yes, no)
famrel - 家族の仲(1~5:低いほど仲が悪く、高いほど仲が良い)
freetime - 放課後の自由時間(1~5:低いほど自由時間が少なく、高いほど自由時間が多い)
goout - 友達と出かける(1~5:低いほど友達と出かける回数が少なく、高いほど出かける回数が多い)
Dalc - 平日のアルコール摂取量(1~5:低いほど平日のアルコール摂取量が少なく、高いほど多い)
Walc - 週末のアルコール摂取量(1~5:低いほど週末のアルコール摂取量が少なく、高いほど多い)
health - 健康状態(1~5:低いほど体調が悪く、高いほど体調が良い)
absences - 不登校数
G1 - 前期の成績
G2 - 後期の成績
G3 - 最終成績
# In[17]:


# 各カラムの名前変更

df = df.rename(columns={'school':'学校', 'sex':'性別', 'address':'地域', 'age':'年齢','Fjob':'父の仕事',
                        'famsize':'家族の大きさ', 'Pstatus':'親の同居状態', 
                        'Medu':'母の教育', 'Fedu':'父の教育', 'Mjob':'母の仕事',
                        'Fedu':'父の仕事', 'reason':'学校を選ぶ理由', 'guardian':'生徒の保護者',
                        'traveltime':'通学時間', 'studytime':'毎週の学習時間', 'failures':'過去のクラスの失敗数',
                        'schoolsup':'追加の教育サポート', 'famsup':'家族の教育サポート', 'paid':'追加有料クラス', 'activities':'課外活動',
                        'nursery':'保育園に通っていた', 'higher':'高等教育を受けていた', 'internet':'家でインターネットを使えるか', 'romantic':'恋人がいる','famrel':'家族の仲',
                        'freetime':'放課後の自由時間', 'goout':'友達と出かける', 'Dalc':'平日のアルコール摂取量', 'Walc':'週末のアルコール摂取量', 'health':'健康状態',
                        'absences':'不登校数', 'G1':'前期の成績', 'G2':'後期の成績', 'G3':'最終成績'})

# 列名一覧取得
print(df.columns.values)


# In[19]:


# 全体の欠損値の合計の認
print(df.isnull().values.sum())


# In[21]:


# ダミー変数の作成
df = pd.get_dummies(df, drop_first=True)
print(df.columns.values)


# In[22]:


# 学習データとテストデータの作成
from sklearn.model_selection import train_test_split

# 目的変数は最終成績
target = '最終成績'

# 説明変数は目的変数以外の全て
X = df.loc[:, df.columns!=target]
# yは目的変数のみ
y = df[target]

# hold_out
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[25]:


# 特徴量選択のための標準化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[37]:


# 学習　予測　評価　
from sklearn.linear_model import Lasso
model = Lasso()

# 学習
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 評価(平均二乗誤差)
mse = np.mean((y_test - y_pred) ** 2)
print(mse)


# In[38]:


# 係数の確認
model.coef_


# 単純に後期の成績が良い生徒は最終成績が良いのがわかった。
# けど、これだと雑すぎるのでもう少し詳しく分析していく。(8/3更新済み)

# In[ ]:





# In[ ]:




