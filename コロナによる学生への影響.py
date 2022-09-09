#!/usr/bin/env python
# coding: utf-8

# ### コロナが学生に与える影響

# コロナによって学生の生活は大きく変わりました。<br>
# 授業はオンライン授業になったり、マスクが必須になったり。<br>
# それらのライフスタイルの変化を、データを通じて理解します。<br>
# 

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# In[5]:


# csvデータをデータフレームとして読み込み
df = pd.read_csv('COVID-19 Survey Student Responses.csv')

# データの先頭5行を表示
df.head()


# In[7]:


# 行数、列数の表示
print(df.shape)


# In[9]:


df.info()


# この時点で分かったことをまとめます。<br>
# 
# ・カテゴリ変数が含まれる列がある<br>
# ・'Rating of Online Class experience'と'Medium for online class'に欠損値が含まれる<br>
# ・float型のデータが5列、int型の列が2列、object型の列が12列ある

# ### 各列の説明

# 次に各列について説明します。<br>
# 
# ID：生徒のID<br>
# Region of residence：居住地域<br>
# Age of Subject：年齢<br>
# Time spent on Online Class：オンラインクラスに費やした時間<br>
# Rating of Online Class experience：オンライン授業の評価<br>
# Medium for online class：どんなデバイスでオンライン授業を受けてるか<br>
# Time spent on self study：独学に費やす時間<br>
# Time spent on fitness：フィットネスに費やす時間<br>
# Time spent on sleep：睡眠に費やす時間<br>
# Time spent on social media：ソーシャルメディアに費やした時間<br>
# Prefered social media platform：優先するソーシャルメディアプラットフォーム<br>
# Time spent on TV：テレビに費やす時間<br>
# Number of meals per day：1日の食事回数<br>
# Change in your weight：体重の変化<br>
# Health issue during lockdown：ロックダウン中の健康状態<br>
# Stress busters：ストレス解消法<br>
# Time utilized：利用時間<br>
# Do you find yourself more connected with your family, close friends , relatives ?：家族や友人との繋がりが強くなったか<br>
# What you miss the most：あなたが最も大切なもの<br>

# ### 探索的データ分析(EDA)と可視化

# #### 1:科目について

# ##### 1.1：住んでる地域について

# In[46]:


# 'Region of residence'の値毎に個数をカウントします。
dict_ = df['Region of residence'].value_counts()

# 値に対しての個数を辞書型にして変数に格納
dict_ = dict_.to_dict()
dict_


# In[55]:


# 描画領域の作成
plt.figure(figsize=(10,10))

'''
plt.pieで円グラフを作成する。
x=dict_.values()でプロットするデータを指定、今回は辞書型の変数の値(values)を指定。
labels=dict_.keys()で何がどのくらいの割合か理解するためのラベルを貼る。
autopct='%1.1f%%'でそれぞれ何％かを表示。
startangle=90で円を90°回転。
lt.legend()が無いとラベルが表示されない。
'''
plt.pie(x=dict_.values(), labels=dict_.keys(), autopct='%1.1f%%', startangle=90)
plt.legend()
plt.show()


# デリーに住んでる人が61.0%<br>
# デリー以外に住んでる人が39.0%です。

# #### 1.2. 年齢別分布

# In[59]:


plt.figure(figsize=(12, 8))
'''
sns.countplotは指定したカラムの値の、
それぞれの個数を可視化してくれる。
x='Age of Subject'で'Age of Subject'列を指定。
data=dfで元のデータフレームを指定
'''
sns.countplot(x='Age of Subject', data=df)
plt.yscale('linear')
plt.xlabel('Age of Subject', weight='bold')
plt.ylabel('Number of Subjects', weight='bold')
plt.show()


# #### 1.3:：オンライン学習に使用するデバイス

# In[70]:


dict_ = df['Medium for online class'].value_counts().to_dict()
plt.figure(figsize=(12,12))
plt.pie(x=dict_.values(), startangle=90,labels=dict_.keys())
plt.legend()
plt.show()


# ほとんどの人がスマホか、デスクトップから接続してますね。

# #### 1.4：時間をどのように過ごしたか

# In[86]:


plt.figure(figsize=(15,10))

# 'Time spent on TV'列がobject型だったので、数値型に変換
df['Time spent on TV'] = df['Time spent on TV'].apply(pd.to_numeric, errors='coerce')

# 「何に時間を費やしたか」という列をまとめて、箱ひげ図として表示
sns.boxplot(data=df[['Time spent on Online Class','Time spent on self study','Time spent on fitness',
                    'Time spent on sleep','Time spent on social media','Time spent on TV']],orient='h')
plt.show()


# 睡眠に使ってる時間が多いですね。<br>
# 中央値が7.5以上でよく寝てると言えます。<br>
# <br>
# あとわかりやすいのは、フィットネスに使う時間が極端に低く見えます。<br>
# コロナの影響で運動する機会が減ったのでしょうか？<br>

# #### 1.5:人気のSNS

# In[87]:


dict_ = df['Prefered social media platform'].value_counts().to_dict()
plt.figure(figsize=(12,12))
plt.pie(x=dict_.values(), startangle=0)
plt.legend(labels=dict_.keys(), loc='upper right',shadow=True, facecolor='lightyellow')
plt.show()


# Instagram、WhatsApp、YouTubeが大半を占めてますね。

# #### 1.6：ストレス解消法について

# In[88]:


plt.figure(figsize=(12,8))
# orderに渡した文字列のリストの順番で表示される。
sns.countplot(x='Stress busters',data=df, order=df['Stress busters'].value_counts().index[:15])
plt.ylabel("Number of Subjects", weight='bold')
plt.xlabel("Stress buster activity", weight='bold')
plt.show()


# 音楽鑑賞とオンラインゲームが多いですね。<br>
# コロナ渦で「室内で出来るストレス解消法」が上がったんでしょうか？<br>
# コロナ渦前と比較してみたいです。<br>

# #### 1.7：最も失ったもの

# In[95]:


plt.figure(figsize=(12,8))
sns.countplot(y='What you miss the most', data=df, order=df['What you miss the most'].value_counts().index[:10])
plt.xlabel("Number of Subjects", weight='bold')
plt.ylabel("What did they miss?", weight='bold')
plt.show()


# 学校や大学、友達や親戚といった回答が多かったです。<br>
# ALLと答えた人は絶望していたと思います。<br>
# 

# ### 健康にどのような影響があったか

# #### 2.1：ロックダウン中の健康問題

# 健康問題を聞いたところ、「YES」か「NO」の回答が得られました。<br>
# 両者は時間の使い方に違いがあるでしょうか？<br>
# また、時間をどう使ったときどのような健康状態になるのでしょうか？<br>

# In[98]:


'''
1つの描画領域で複数のグラフを表示します。
plt.subplotの引数で、3行2列のグラフ領域を作ります。
'''
fig, ax=plt.subplots(3, 2, figsize=(16,18))

'''
横軸が「健康状態は？」に対しての回答で、横軸が「～に割いてる時間」とします。
ax=ax[x,y]でどの位置にグラフを表示するか指定します。
'''
sns.violinplot(x='Health issue during lockdown', y='Time spent on Online Class', data=df, ax=ax[0,0])
sns.violinplot(x='Health issue during lockdown', y='Time spent on self study', data=df, ax=ax[0,1])
sns.violinplot(x='Health issue during lockdown', y='Time spent on fitness', data=df, ax=ax[1,0])
sns.violinplot(x='Health issue during lockdown', y='Time spent on sleep', data=df, ax=ax[1,1])
sns.violinplot(x='Health issue during lockdown', y='Time spent on social media', data=df, ax=ax[2,0])
sns.violinplot(x='Health issue during lockdown', y='Time spent on TV', data=df, ax=ax[2,1])
plt.show()


# sns.violinplotは分布密度をグラフにしてくれます。<br>
# これは「相対的な出やすさ」をみえるグラフです。<br>
# 例えば'Health issue during lockdown'と'Time spent on Online Class'のグラフを見てみます。<br>
# NOと答えたの中だと'Time spent on Online Class'が4のところが膨らんでます。<br>
# つまり、NOと答えた人の中で、'Time spent on Online Class'が4と答えた人が最も多い、ということが分かります。<br>
# 逆に、YESと答えた人は'Time spent on Online Class'が4の割合は低いですね。<br>
# つまり、健康状態が良い人はオンライン授業に費やした時間が、NOと答えた人より全体的に高いと言えます。<br>
# ただ、他のグラフも大体同じような形をしてますね。
# 

# #### 2.2年齢について
# 

# In[99]:


plt.figure(figsize=(10,8))
sns.violinplot(y='Health issue during lockdown', x='Age of Subject', data=df)
plt.show()


# 健康では無いと答えた人は、健康だと言って人に比べて年齢が高い人が多いです。

# #### 2.3食事数について
# 

# In[101]:


plt.figure(figsize=(10,8))
sns.set(style='darkgrid')
sns.violinplot(y='Health issue during lockdown', x='Number of meals per day', data=df)
plt.show()


# どちらも2回か3回と答えてる人の割合が多いと言えます。<br>
# 4回以降と答えた人の割合は、NOと答えた人の方が高いですね。

# ### 3：食事の回数は体重に影響しますか？
# 

# In[102]:


plt.figure(figsize=(12,8))
sns.boxplot(y='Change in your weight', x='Number of meals per day',data=df)
plt.show()


# In[129]:


for i in df['Change in your weight'].value_counts().index:
    if i == 'Remain Constant':
        print(i,'は体重に変化無し')
    elif i == 'Increased':
        print(i,'は体重が増加した')
    else:
        print(i,'は体重が減少した')


# 体重が増加した人は食事回数が多いように見えますね。<br>
# しかし、体重が減少した人と変化しなかった人の箱ひげ図はあまり変わらないように思います。<br>
# (ただ、母数が違うことを考慮しないといけませんが)

# ### 4:時間をどのように活用したか
# 

# In[131]:


fig, ax = plt.subplots(3,2, figsize=(16,18))
sns.boxplot(x='Time utilized', y='Time spent on Online Class', data=df, ax=ax[0,0])
sns.boxplot(x='Time utilized', y='Time spent on self study', data=df, ax=ax[0,1])
sns.boxplot(x='Time utilized', y='Time spent on fitness', data=df, ax=ax[1,0])
sns.boxplot(x='Time utilized', y='Time spent on sleep', data=df, ax=ax[1,1])
sns.boxplot(x='Time utilized', y='Time spent on social media', data=df, ax=ax[2,0])
sns.boxplot(x='Time utilized', y='Time spent on TV', data=df, ax=ax[2,1])
plt.show()


# ### まとめ

# In[ ]:




