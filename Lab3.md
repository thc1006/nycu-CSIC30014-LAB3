### **第一頁: 封面頁**

**擷取文字:**

Lab3 Multi-class Classification  
TA: Frank

頁面分析:  
此為簡報的封面頁，明確點出本次實驗的主題為「多類別分類」，並告知授課助教 (TA) 是 Frank。

### **第二頁: 重要規則**

**擷取文字:**

Important Rule  
Submission Deadline: 11/17 (Mon.) 12:30 pm  
Late report submission: grade \* 80%, note that the Kaggle leaderboard has no late submission  
Turn in:  
1\) Experiment Report (.pdf) to E3 「LAB3\_yourstudentID\_name.pdf eg: 「LAB3\_313XXXXXX\_王小翔.pdf」  
2\) Source Code (.py) to your own github (No need to upload to E3)

頁面分析:  
本頁說明了重要的繳交規則：

1. **繳交截止日期:** 11月17日 (星期一) 中午 12:30。  
2. **遲交罰則:** 實驗報告若遲交，分數將乘以 80% (打八折)。特別強調 Kaggle 競賽平台沒有遲交選項。  
3. **繳交項目與方式:**  
   * **實驗報告:** 需繳交 PDF 檔至 E3 平台，檔名格式為「LAB3\_學號\_姓名.pdf」，並舉例「LAB3\_313XXXXXX\_王小翔.pdf」。  
   * **原始碼:** 需上傳 .py 檔案至自己的 GitHub，不需上傳至 E3 平台。

### **第三頁: 實驗目標**

**擷取文字:**

Lab Objective  
Multi-class Classification with CXR dataset  
Join the Kaggle webset is necessary\!  
URL: \[https://www.kaggle.com/t/ff9cf59609724a9fb396a5ef8557013b\](https://www.kaggle.com/t/ff9cf59609724a9fb396a5ef8557013b)  
Class: Normal, Bacteria (Pneumonia), Virus (Pneumonia), COVID-19  
(Predicted answer will only belong to a class)  
Object 1: Multi-class classification  
Object 2: Achieve the highest ranking as you can (Evaluation metric: macro-F1)  
Object 3: Finish the report according to the spec

頁面分析:  
本頁闡述了實驗的核心目標：

1. **任務:** 使用 CXR (胸腔 X 光) 資料集進行多類別分類。  
2. **平台:** 強制要求必須參加指定的 Kaggle 競賽，並提供了競賽的網址。  
3. **分類類別:** 總共有四個類別：正常 (Normal)、細菌性肺炎 (Bacteria Pneumonia)、病毒性肺炎 (Virus Pneumonia)、新冠肺炎 (COVID-19)。預測結果必須是這四者之一。  
4. **三大目標:**  
   * 目標一：完成多類別分類任務。  
   * 目標二：在 Kaggle 競賽中盡可能取得高排名，評分指標為 macro-F1 分數。  
   * 目標三：依照規格完成實驗報告。

### **第四頁: Kaggle 競賽 (結果格式)**

**擷取文字:**

Kaggle Competition  
• URL: \[https://www.kaggle.com/t/ff9cf59609724a9fb396a5ef8557013b\](https://www.kaggle.com/t/ff9cf59609724a9fb396a5ef8557013b)  
• Result Format (Evaluate the test\_images)  
  Ο The result should be a csv file (refer to the test\_data\_sample.csv)  
  Ο The file should contain the new\_filename and the predicted result of each class, do not change the order of filename  
Example:

圖片分析:  
頁面中附有一張 CSV 檔案的範例截圖。

* **圖片內容:** 該截圖展示了提交結果的 CSV 檔案格式。  
* **欄位結構:**  
  * 第一欄是索引值。  
  * A 欄標題為 new\_filename，內容是測試圖片的檔名 (例如 1736.jpeg)。  
  * B 欄標題為 normal。  
  * C 欄標題為 bacteria。  
  * D 欄標題為 virus。  
  * E 欄標題為 COVID-19。  
* **數值意義:** 採用 one-hot encoding 格式，模型預測的類別在對應欄位標示為 1，其餘類別標示為 0。例如，第二行 (1736.jpeg) 的 normal 欄位為 1，表示模型預測這張圖片為「正常」。  
* **整體分析:** 此頁面詳細說明了 Kaggle 競賽的繳交檔案格式。必須是一個 CSV 檔案，包含檔名以及對應四個類別的預測結果，且不能更改檔名的順序。

### **第五頁: Kaggle 競賽 (團隊名稱)**

**擷取文字:**

Kaggle Competition  
• The final result metric is macro-F1  
• Please set your team name as your student ID. Otherwise, you'll get no points in the competition  
• Note that the max result submit times per day is 10 times

圖片分析:  
頁面中附有一張 Kaggle 網站的介面截圖。

* **圖片內容:** 截圖顯示了 Kaggle 競賽中設定「團隊名稱 (TEAM NAME)」的區塊。  
* **介面元素:**  
  * 標題為「Your Team」。  
  * 輸入框中顯示了範例隊名「Frank LinYJ」。  
  * 輸入框下方有提示文字：「This name will appear on your team's leaderboard position」(此名稱將會顯示在您團隊的排行榜位置上)。  
  * 上方可見 Kaggle 競賽的導覽列，包含 Overview, Data, Code, Models, Discussion, Leaderboard, Rules, Team, Submissions 等分頁。  
* **整體分析:** 此頁面補充了 Kaggle 競賽的重要規則：  
  1. 最終的評分指標是 macro-F1。  
  2. **團隊名稱必須設定為學號**，否則競賽部分將不計分。  
  3. 每天最多有 10 次提交結果的機會。

### **第六頁: 資料**

**擷取文字:**

Data  
• The dataset contains (4017+709+1182) images  
  Ο Train: 4017  
  Ο Val: 709  
  Ο Test: 1182  
• If you do not need the validation data, you could combine it with training data  
• Train distribution:  
  normal 0.266866/ bacteria 0.470002/virus 0.253423/COVID-19 0.009709  
• Val distribution:  
  Ο normal 0.266573/ bacteria 0.469676 / virus 0.253879/COVID-19 0.009873  
• Test distribution:  
  Ο normal 0.266497/ bacteria 0.470389 / virus 0.252961/ COVID-19 0.010152

頁面分析:  
本頁詳細介紹了本次實驗所使用的資料集：

1. **資料集大小:**  
   * 訓練集 (Train): 4017 張圖片  
   * 驗證集 (Val): 709 張圖片  
   * 測試集 (Test): 1182 張圖片  
2. **使用建議:** 如果不需要使用驗證集，可以將其與訓練集合併。  
3. **類別分佈:**  
   * 提供了訓練、驗證、測試三個資料集中，四個類別的精確比例。  
   * 可以觀察到嚴重的**類別不平衡**問題：「bacteria」類別佔比最高 (約 47%)，而「COVID-19」類別佔比極低 (約 1%)。這將是模型訓練時需要解決的一大挑戰。  
   * 三個資料集的類別分佈非常相似。

### **第七頁: 需求**

**擷取文字:**

Requirements  
1\. Implement any model to do the task is allowed, including calling API.  
2\. Write your own training code.  
3\. Discuss all your experiments and your findings in the report's discussion section. The more variations you explore, the better your grade.  
   • The testing result can refer to the leaderboard  
4\. Upload you code to your own github including README.md

頁面分析:  
此頁列出了完成實驗的具體要求：

1. **模型自由度:** 允許使用任何模型來完成任務，甚至可以呼叫 API。  
2. **程式碼撰寫:** 必須自行撰寫訓練模型的程式碼。  
3. **實驗與討論:** 報告的討論區塊需要詳述所有進行過的實驗與發現。探索越多的變化，分數會越高。測試結果可以直接參考 Kaggle 排行榜。  
4. **程式碼繳交:** 需將程式碼連同 README.md 說明檔案上傳至自己的 GitHub。

### **第八頁: 報告規格 (一)**

**擷取文字:**

Report Spec  
1\. Introduction (5%)  
   a. Introduce the task  
2\. Implementation details (20%)  
   a. The details of your model (including settings and introduce ur model)  
   b. The details of your Dataloader (mainly the data augmentation strategies)  
3\. Strategy design (50%) (Most important part)  
   a. How did you pre-process your data? (histogram equalization, center cropping ...)  
   b. What makes your training strategy special?  
      i. model design, framework design, loss function design ...  
   c. All of your training details  
      i. hyperparameters, settings, ...

頁面分析:  
本頁開始詳細說明實驗報告的結構與各部分的配分：

1. **介紹 (5%):** 簡介任務內容。  
2. **實作細節 (20%):**  
   * 模型的詳細資訊 (包含設定與模型介紹)。  
   * Dataloader 的細節 (主要著重在資料增強策略)。  
3. **策略設計 (50%) \- 最重要的部分:**  
   * 資料預處理的方法 (如：直方圖均衡化、中心裁剪等)。  
   * 訓練策略的特殊之處 (如：模型設計、框架設計、損失函數設計等)。  
   * 所有訓練的細節 (如：超參數、各種設定等)。

### **第九頁: 報告規格 (二)**

**擷取文字:**

Report Spec  
4\. Discussion (20%)  
   a. Discuss your findings or share anything you want to share  
5\. Github Link (5%) (Do not forget)

頁面分析:  
此頁接續報告規格的說明：  
4\. 討論 (20%): 討論你的發現，或分享任何想分享的心得。  
5\. GitHub 連結 (5%): 在報告中附上 GitHub 連結，並提醒不要忘記。

### **第十頁: 分數標準**

**擷取文字:**

Score criterion of Lab3  
Score: 30% Performance \+ 70% Report  
P.S If the report exists format errors (file name or the report spec), it will be 5 points penalty (-5)  
Criterion of result (30%)  
Top1: 30 pts  
Top 25%: 25 pts (No. 2-4)  
Top 50%: 20 pts (No. 5-7)  
Rest: 15 pts

頁面分析:  
最後一頁說明了整體的評分標準：

1. **總分組成:** 最終分數由 30% 的 Kaggle 表現和 70% 的報告內容組成。  
2. **格式錯誤罰則:** 如果報告的檔名或內容規格不符，將會扣 5 分。  
3. **表現分數 (30%) 標準:**  
   * 第 1 名: 獲得 30 分。  
   * 前 25% (第 2-4 名): 獲得 25 分。  
   * 前 50% (第 5-7 名): 獲得 20 分。  
   * 其餘名次: 獲得 15 分。  
     (註: 根據括號內的名次推斷，總參與人數可能約為 14 人)