# **Page 1**

Lab3 Multi-class Classification  
TA: Frank

# **Page 2**

**Important Rule**

* **Competition Deadline:** 11/17 (Mon.) 12:30 pm  
* **Submission Deadline:** 11/21 (Fri.) 12:30 pm  
* **Late report submission:** grade \* 80%, note that the Kaggle leaderboard has no late submission

**Turn in:**

1. **Experiment Report (.pdf)** to E3  
   * Naming format: 「LAB3\_yourstudentID\_name.pdf」  
   * eg: 「LAB3\_313XXXXXX\_王小翔.pdf」  
2. **Source Code (.py)** to your own github (No need to upload to E3)

# **Page 3**

**Lab Objective**

Multi-class Classification with CXR dataset

**Join the Kaggle competition is necessary\!**

* **URL:** https://www.kaggle.com/t/ff9cf59609724a9fb396a5ef8557013b  
* **Class:** Normal, Bacteria (Pneumonia), Virus (Pneumonia), COVID-19

(Predicted answer will only belong to a class)

* **Object 1:** Multi-class classification  
* **Object 2:** Achieve the highest ranking as you can (Evaluation metric: macro-F1)  
* **Object 3:** Finish the report according to the spec

# **Page 4**

**Kaggle Competition**

* **URL:** https://www.kaggle.com/t/ff9cf59609724a9fb396a5ef8557013b  
* **Result Format (Evaluate the test\_images)**  
  * The result should be a csv file (refer to the test\_data\_sample.csv)  
  * The file should contain the new\_filename and the predicted result of each class, do not change the order of filename

\[圖片/表格資訊分析：CSV 格式範例\]  
這頁包含一個 CSV 檔案結構的示意圖，展示了提交檔案必須具備的欄位與 One-hot encoding 格式：

| (Row) | A (new filename) | B (normal) | C (bacteria) | D (virus) | E (CoVID19-) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | new filename | normal | bacteria | virus | CoVID19- |
| 2 | 1736.jpeg | 1 | 0 | 0 | 0 |
| 3 | 3397.jpeg | 1 | 0 | 0 | 0 |
| 4 | 1446.jpeg | 1 | 0 | 0 | 0 |
| 5 | 4619.jpeg | 1 | 0 | 0 | 0 |
| 6 | 5639.jpeg | 1 | 0 | 0 | 0 |
| 7 | 956.jpeg | 1 | 0 | 0 | 0 |
| 8 | 3376.jpeg | 0 | 0 | 1 | 0 |
| 9 | 4151.jpeg | 0 | 1 | 0 | 0 |
| 10 | 676.jpeg | 1 | 0 | 0 | 0 |
| 11 | 4821.jpeg | 0 | 1 | 0 | 0 |
| 12 | 632.jpeg | 1 | 0 | 0 | 0 |
| 13 | 1263.jpeg | 1 | 0 | 0 | 0 |
| 14 | 2044.jpeg | 0 | 1 | 0 | 0 |
| 15 | 1819.jpeg | 0 | 1 | 0 | 0 |

# **Page 5**

**Kaggle Competition**

* The evaluation metric is **macro-F1**  
* Please set your team name as your **student ID**. Otherwise, you'll get no points in the competition (0 point in the performance score).  
* Note that the max result submit times per day is **10 times**

\[圖片資訊分析：Kaggle 介面截圖\]  
圖片顯示了 Kaggle 比賽的介面選單與隊伍設定方式：

1. **Menu Bar:** Overview, Data, Code, Models, Discussion, Leaderboard, Rules, Team, Submissions  
2. **Your Team Section:** "Everyone that competes in a Competiton does so as a team even if you're competing by yourself. Learn more."  
3. **General Section:**  
   * **TEAM NAME:** 欄位中填寫了範例 "Frank LinYJ" (注意：學生應填寫學號)  
   * 提示文字: "This name will appear on your team's leaderboard position"

# **Page 6**

**Data**

* The dataset contains **(4017 \+ 709 \+ 1182\)** images  
  * **Train:** 4017  
  * **Val:** 709  
  * **Test:** 1182  
* If you do not need the validation data, you could combine it with training data

**Distributions:**

* **Train distribution:**  
  * normal: 0.266866  
  * bacteria: 0.470002  
  * virus: 0.253423  
  * COVID-19: 0.009709  
* **Val distribution:**  
  * normal: 0.266573  
  * bacteria: 0.469676  
  * virus: 0.253879  
  * COVID-19: 0.009873  
* **Test distribution:**  
  * normal: 0.266497  
  * bacteria: 0.470389  
  * virus: 0.252961  
  * COVID-19: 0.010152

# **Page 7**

**Requirements**

1. **Implement any model** to do the task is allowed, including calling API.  
2. **Write your own training code.**  
3. **Discuss all your experiments and your findings** in the discussion section.  
   * The more variations you explore, the better your grade.  
   * The testing result can refer to the leaderboard.  
4. **Upload your code to your own github** including README.md.

# **Page 8**

**Report Spec**

1. Introduction (5%)  
   a. Introduce the task  
2. Implementation details (20%)  
   a. The details of your model (including settings and introduce ur model)  
   b. The details of your Dataloader (mainly the data augmentation strategies)  
3. Strategy design (50%) (Most important part)  
   a. How did you pre-process your data? (histogram equalization, center cropping ...)  
   b. What makes your training strategy special?  
   i. model design, framework design, loss function design ...  
   c. All of your training details  
   i. hyperparameters, settings, ...

# **Page 9**

**Report Spec**

4. Discussion (20%)  
   a. Discuss your findings or share anything you want to share  
5. **Github Link (5%)** (Do not forget)

# **Page 10**

**Score criterion of Lab3**

**Score: 30% Performance \+ 70% Report**

P.S If the report exists format errors (file name or the report spec), it will be **5 points penalty (-5)**

**Criterion of result (30%)**

* **Top 1:** 30 pts  
* **Top 25%:** 25 pts (No. 2-4)  
* **Top 50%:** 20 pts (No. 5-7)  
* **Top 75%:** 18 pts (No. 8-10)  
* **Rest:** 15 pts