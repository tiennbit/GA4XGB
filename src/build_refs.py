# -*- coding: utf-8 -*-
"""Chuyển citation key [author2024-slug] trong draft thành số IEEE [1],[2]... và
sinh danh sách References đúng định dạng IEEE.

Đánh số theo THỨ TỰ XUẤT HIỆN trong bài (quy tắc IEEE).
Chỉ dùng các ref đã verify — key nào không có trong BIB sẽ báo lỗi, KHÔNG bịa.

Chạy: python3 src/build_refs.py
Xuất: paper/draft_numbered.md (bản đã đánh số + có mục References)
"""
import re
import sys

DRAFT = "paper/draft_v1.md"
OUT = "paper/draft_numbered.md"

# Thư mục đã xác minh (nguồn: paper/refs_verified.md — mọi mục đã qua Crossref/OpenAlex)
BIB = {
"bischl2023-hpo": 'B. Bischl et al., "Hyperparameter optimization: Foundations, algorithms, best practices, and open challenges," <i>WIREs Data Mining Knowl. Discovery</i>, vol. 13, no. 2, Mar. 2023, Art. no. e1484, doi: 10.1002/widm.1484.',
"aghaabbasi2023-bo": 'M. Aghaabbasi, M. Ali, M. Jasiński, Z. Leonowicz, and T. Novák, "On hyperparameter optimization of machine learning methods using a Bayesian optimization algorithm to predict work travel mode choice," <i>IEEE Access</i>, vol. 11, pp. 19762–19774, 2023, doi: 10.1109/ACCESS.2023.3247448.',
"hoque2021-tuning": 'K. E. Hoque and H. Aljamaan, "Impact of hyperparameter tuning on machine learning models in stock price forecasting," <i>IEEE Access</i>, vol. 9, pp. 163815–163830, 2021, doi: 10.1109/ACCESS.2021.3134138.',
"elgeldawi2021-comparative": 'E. Elgeldawi, A. Sayed, A. R. Galal, and A. M. Zaki, "Hyperparameter tuning for machine learning algorithms used for Arabic sentiment analysis," <i>Informatics</i>, vol. 8, no. 4, Nov. 2021, Art. no. 79, doi: 10.3390/informatics8040079.',
"moraleshernandez2023-survey": 'A. Morales-Hernández, I. Van Nieuwenhuyse, and S. Rojas Gonzalez, "A survey on multi-objective hyperparameter optimization algorithms for machine learning," <i>Artif. Intell. Rev.</i>, vol. 56, no. 8, pp. 8043–8093, Aug. 2023, doi: 10.1007/s10462-022-10359-2.',
"ghatasheh2022-gaxgb": 'N. Ghatasheh, I. Altaharwa, and K. Aldebei, "Modified genetic algorithm for feature selection and hyper parameter optimization: Case of XGBoost in spam prediction," <i>IEEE Access</i>, vol. 10, pp. 84365–84383, 2022, doi: 10.1109/ACCESS.2022.3196905.',
"ghatasheh2023-telemarketing": 'N. Ghatasheh, I. Altaharwa, and K. Aldebei, "Modeling the telemarketing process using genetic algorithms and extreme boosting: Feature selection and cost-sensitive analytical approach," <i>IEEE Access</i>, vol. 11, pp. 67806–67824, 2023, doi: 10.1109/ACCESS.2023.3292840.',
"kim2024-gasurrogate": 'C. Kim and I. Joe, "A balanced approach of rapid genetic exploration and surrogate exploitation for hyperparameter optimization," <i>IEEE Access</i>, vol. 12, pp. 192184–192194, 2024, doi: 10.1109/ACCESS.2024.3508269.',
"emara2025-apso": 'A. M. Emara, G. Atteia, and J. H. Alkhateeb, "Fine tuning hyperparameters of deep learning models using metaheuristic accelerated particle swarm optimization algorithm," <i>IEEE Access</i>, vol. 13, pp. 134506–134518, 2025, doi: 10.1109/ACCESS.2025.3591403.',
"kamaldas2025-gwo": 'M. Kamal Das, C. Columbus Chinnappan, and E. Elakiya, "A temporal attention-based SARIMA-BiLSTM residual learning model tuned by grey wolf optimizer for parallel urban traffic forecasting," <i>IEEE Access</i>, vol. 13, pp. 136073–136086, 2025, doi: 10.1109/ACCESS.2025.3590104.',
"mahadeva2022-gwoann": 'R. Mahadeva, M. Kumar, S. P. Patole, and G. Manik, "Desalination plant performance prediction model using grey wolf optimizer based ANN approach," <i>IEEE Access</i>, vol. 10, pp. 34550–34561, 2022, doi: 10.1109/ACCESS.2022.3162932.',
"mumtahina2024-review": 'U. Mumtahina, S. Alahakoon, and P. Wolfs, "Hyperparameter tuning of load-forecasting models using metaheuristic optimization algorithms—A systematic review," <i>Mathematics</i>, vol. 12, no. 21, Oct. 2024, Art. no. 3353, doi: 10.3390/math12213353.',
"mehdary2024-gaxgb": 'A. Mehdary, A. Chehri, A. Jakimi, and R. Saadane, "Hyperparameter optimization with genetic algorithms and XGBoost: A step forward in smart grid fraud detection," <i>Sensors</i>, vol. 24, no. 4, Feb. 2024, Art. no. 1230, doi: 10.3390/s24041230.',
"yokoyama2024-mohpo": 'A. M. Yokoyama, M. Ferro, and B. Schulze, "Multi-objective hyperparameter optimization approach with genetic algorithms towards efficient and environmentally friendly machine learning," <i>AI Commun.</i>, vol. 37, no. 3, pp. 429–442, 2024, doi: 10.3233/AIC-230063.',
"nguyen2021-waterlevel": 'D. H. Nguyen, X. H. Le, J.-Y. Heo, and D.-H. Bae, "Development of an extreme gradient boosting model integrated with evolutionary algorithms for hourly water level prediction," <i>IEEE Access</i>, vol. 9, pp. 125853–125867, 2021, doi: 10.1109/ACCESS.2021.3111287.',
"ainan2024-bankruptcy": 'U. H. Ainan, L. Y. Por, Y.-L. Chen, J. Yang, and C. S. Ku, "Advancing bankruptcy forecasting with hybrid machine learning techniques: Insights from an unbalanced Polish dataset," <i>IEEE Access</i>, vol. 12, pp. 9369–9381, 2024, doi: 10.1109/ACCESS.2024.3354173.',
"li2024-diabetes": 'W. Li, Y. Peng, and K. Peng, "Diabetes prediction model based on GA-XGBoost and stacking ensemble algorithm," <i>PLoS ONE</i>, vol. 19, no. 9, Sep. 2024, Art. no. e0311222, doi: 10.1371/journal.pone.0311222.',
"huang2025-permeability": 'C. Huang, X. Zhu, M. Lu, Y. Zhang, and S. Yang, "XGBoost algorithm optimized by simulated annealing genetic algrithm for permeability prediction modeling of carbonate reservoirs," <i>Sci. Rep.</i>, vol. 15, 2025, Art. no. 14882, doi: 10.1038/s41598-025-99627-z.',
"nguyen2025-ga4rf": 'H.-N. Nguyen, H.-L. Le, N.-T.-T. Trang, and D.-N. Nguyen, "GA4RF: An effective fall detection system through optimizing random forest hyperparameters using genetic algorithm with mobile sensor data," <i>IEEE Access</i>, vol. 13, pp. 139802–139815, 2025, doi: 10.1109/ACCESS.2025.3596520.',
"steininger2021-denseweight": 'M. Steininger, K. Kobs, P. Davidson, A. Krause, and A. Hotho, "Density-based weighting for imbalanced regression," <i>Mach. Learn.</i>, vol. 110, no. 8, pp. 2187–2211, Aug. 2021, doi: 10.1007/s10994-021-06023-5.',
"yang2021-dir": 'Y. Yang, K. Zha, Y.-C. Chen, H. Wang, and D. Katabi, "Delving into deep imbalanced regression," in <i>Proc. 38th Int. Conf. Mach. Learn. (ICML)</i>, in Proceedings of Machine Learning Research, vol. 139, 2021, pp. 11842–11851.',
"avelino2024-resampling": 'J. G. Avelino, G. D. C. Cavalcanti, and R. M. O. Cruz, "Resampling strategies for imbalanced regression: A survey and empirical analysis," <i>Artif. Intell. Rev.</i>, vol. 57, no. 4, Mar. 2024, Art. no. 82, doi: 10.1007/s10462-024-10724-3.',
"in2025-drf": 'D. D. In and H. Kim, "Distance-based relevance function for imbalanced regression," <i>Stats</i>, vol. 8, no. 3, Jun. 2025, Art. no. 53, doi: 10.3390/stats8030053.',
"shahbazi2026-hybrid": 'S. Shahbazi, H. Mohammadi, and M. Afsharchi, "Hybrid imbalanced regression through unified data-level and algorithm-level balancing," <i>Expert Syst. Appl.</i>, vol. 322, 2026, Art. no. 131908, doi: 10.1016/j.eswa.2026.131908.',
"puetz2026-deconstructing": 'N. C. Puetz, J. U. Brandt, M. Hilbert, E. Raponi, T. Bäck, and T. Bartz-Beielstein, "Deconstructing deep imbalanced regression: A comprehensive review and experimental evaluation," <i>Artif. Intell. Rev.</i>, vol. 59, no. 6, 2026, Art. no. 135, doi: 10.1007/s10462-026-11570-1.',
"limamarinho2024-metalearning": 'T. Lima Marinho, D. C. do Nascimento, and B. A. Pimentel, "Optimization on selecting XGBoost hyperparameters using meta-learning," <i>Expert Syst.</i>, vol. 41, no. 9, Sep. 2024, Art. no. e13611, doi: 10.1111/exsy.13611.',
"elshawi2025-totune": 'R. El Shawi, M. Bahmani, and S. Sakr, "To tune or not to tune? An approach for recommending important hyperparameters for classification and clustering algorithms," <i>Future Gener. Comput. Syst.</i>, vol. 163, 2025, Art. no. 107524, doi: 10.1016/j.future.2024.107524.',
"vasquezramos2025-rsm": 'J. Vasquez-Ramos et al., "Response surface-driven hyperparameter optimization for XGBoost," <i>J. Supercomput.</i>, vol. 81, no. 10, 2025, Art. no. 1112, doi: 10.1007/s11227-025-07600-4.',
"boldini2023-guidelines": 'D. Boldini, F. Grisoni, D. Kuhn, L. Friedrich, and S. A. Sieber, "Practical guidelines for the use of gradient boosting for molecular property prediction," <i>J. Cheminformatics</i>, vol. 15, 2023, Art. no. 73, doi: 10.1186/s13321-023-00743-7.',
"farhadpour2024-macro": 'S. Farhadpour, T. A. Warner, and A. E. Maxwell, "Selecting and interpreting multiclass loss and accuracy assessment metrics for classifications with class imbalance: Guidance and best practices," <i>Remote Sens.</i>, vol. 16, no. 3, Jan. 2024, Art. no. 533, doi: 10.3390/rs16030533.',
"altalhan2025-imbalanced": 'M. Altalhan, A. Algarni, and M. Turki-Hadj Alouane, "Imbalanced data problem in machine learning: A review," <i>IEEE Access</i>, vol. 13, pp. 13686–13699, 2025, doi: 10.1109/ACCESS.2025.3531662.',
"opitz2024-metrics": 'J. Opitz, "A closer look at classification evaluation metrics and a critical reflection of common evaluation practice," <i>Trans. Assoc. Comput. Linguistics</i>, vol. 12, pp. 820–836, 2024, doi: 10.1162/tacl_a_00675.',
"chen2016-xgboost": 'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining</i>, Aug. 2016, pp. 785–794, doi: 10.1145/2939672.2939785.',
"sahlaoui2021-shap": 'H. Sahlaoui, E. A. A. Alaoui, A. Nayyar, S. Agoujil, and M. M. Jaber, "Predicting and interpreting student performance using ensemble models and Shapley additive explanations," <i>IEEE Access</i>, vol. 9, pp. 152688–152703, 2021, doi: 10.1109/ACCESS.2021.3124270.',
"bujang2021-multiclass": 'S. D. Abdul Bujang et al., "Multiclass prediction model for student grade prediction using machine learning," <i>IEEE Access</i>, vol. 9, pp. 95608–95621, 2021, doi: 10.1109/ACCESS.2021.3093563.',
"sun2023-multifeature": 'D. Sun et al., "A university student performance prediction model and experiment based on multi-feature fusion and attention mechanism," <i>IEEE Access</i>, vol. 11, pp. 112307–112319, 2023, doi: 10.1109/ACCESS.2023.3323365.',
"chen2023-edm": 'Z. Chen, G. Cen, Y. Wei, and Z. Li, "Student performance prediction approach based on educational data mining," <i>IEEE Access</i>, vol. 11, pp. 131260–131272, 2023, doi: 10.1109/ACCESS.2023.3335985.',
"raji2024-deaf": 'N. R. Raji, R. M. S. Kumar, and C. L. Biji, "Explainable machine learning prediction for the academic performance of deaf scholars," <i>IEEE Access</i>, vol. 12, pp. 23595–23612, 2024, doi: 10.1109/ACCESS.2024.3363634.',
"nguyenhuy2022-copula": 'T. Nguyen-Huy et al., "Student performance predictions for advanced engineering mathematics course with new multivariate copula models," <i>IEEE Access</i>, vol. 10, pp. 45112–45136, 2022, doi: 10.1109/ACCESS.2022.3168322.',
"yagci2022-edm": 'M. Yağcı, "Educational data mining: Prediction of students\' academic performance using machine learning algorithms," <i>Smart Learn. Environments</i>, vol. 9, no. 1, Mar. 2022, Art. no. 11, doi: 10.1186/s40561-022-00192-z.',
"rodriguez2021-ann": 'C. F. Rodríguez-Hernández, M. Musso, E. Kyndt, and E. Cascallar, "Artificial neural networks in academic performance prediction: Systematic implementation and predictor evaluation," <i>Comput. Educ. Artif. Intell.</i>, vol. 2, 2021, Art. no. 100018, doi: 10.1016/j.caeai.2021.100018.',
"mastour2023-highstakes": 'H. Mastour, T. Dehghani, E. Moradi, and S. Eslami, "Early prediction of medical students\' performance in high-stakes examinations using machine learning approaches," <i>Heliyon</i>, vol. 9, no. 7, Jul. 2023, Art. no. e18248, doi: 10.1016/j.heliyon.2023.e18248.',
"kaensar2023-admission": 'C. Kaensar and W. Wongnin, "Predicting new student performances and identifying important attributes of admission data using machine learning techniques with hyperparameter tuning," <i>Eurasia J. Math. Sci. Technol. Educ.</i>, vol. 19, no. 12, 2023, Art. no. em2369, doi: 10.29333/ejmste/13863.',
"son2022-admission": 'N. T. K. Son, N. V. Bien, N. H. Quynh, and C. C. Tho, "Machine learning based admission data processing for early forecasting students\' learning outcomes," <i>Int. J. Data Warehousing Mining</i>, vol. 18, no. 1, pp. 1–15, 2022, doi: 10.4018/IJDWM.313585.',
"chen2022-gaokao": 'X. D. Chen, P. Yi, Y.-C. Gao, and S.-M. Cai, "A competition model for prediction of admission scores of colleges and universities in Chinese college entrance examination," <i>PLoS ONE</i>, vol. 17, no. 10, Oct. 2022, Art. no. e0274221, doi: 10.1371/journal.pone.0274221.',
"wyness2023-grade-expectations": 'G. Wyness, L. Macmillan, J. Anders, and C. Dilnot, "Grade expectations: How well can past performance predict future grades?" <i>Educ. Econ.</i>, vol. 31, no. 4, pp. 397–418, 2023, doi: 10.1080/09645292.2022.2113861.',
"denes2023-gcse": 'G. Dénes, "A case study of using AI for General Certificate of Secondary Education (GCSE) grade prediction in a selective independent school in England," <i>Comput. Educ. Artif. Intell.</i>, vol. 4, 2023, Art. no. 100129, doi: 10.1016/j.caeai.2023.100129.',
"adnan2021-atrisk": 'M. Adnan et al., "Predicting at-risk students at different percentages of course length for early intervention using machine learning models," <i>IEEE Access</i>, vol. 9, pp. 7519–7539, 2021, doi: 10.1109/ACCESS.2021.3049446.',
"pek2023-atrisk": 'R. Z. Pek, S. T. Özyer, T. Elhage, T. Özyer, and R. Alhajj, "The role of machine learning in identifying students at-risk and minimizing failure," <i>IEEE Access</i>, vol. 11, pp. 1224–1243, 2023, doi: 10.1109/ACCESS.2022.3232984.',
"skittou2024-ews": 'M. Skittou, M. Merrouchi, and T. Gadi, "Development of an early warning system to support educational planning process by identifying at-risk students," <i>IEEE Access</i>, vol. 12, pp. 2260–2271, 2024, doi: 10.1109/ACCESS.2023.3348091.',
"limanto2024-glowsmote": 'S. Limanto, J. L. Buliali, and A. Saikhu, "GLoW SMOTE-D: Oversampling technique to improve prediction model performance of students failure in courses," <i>IEEE Access</i>, vol. 12, pp. 8889–8901, 2024, doi: 10.1109/ACCESS.2024.3351569.',
"pelima2024-slr": 'L. R. Pelima, Y. Sukmana, and Y. Rosmansyah, "Predicting university student graduation using academic performance and machine learning: A systematic literature review," <i>IEEE Access</i>, vol. 12, pp. 23451–23465, 2024, doi: 10.1109/ACCESS.2024.3361479.',
"nabil2021-dnn": 'A. Nabil, M. Seyam, and A. AbouElfetouh, "Prediction of students\' academic performance based on courses\' grades using deep neural networks," <i>IEEE Access</i>, vol. 9, pp. 140731–140746, 2021, doi: 10.1109/ACCESS.2021.3119596.',
"kusumawardani2023-transformer": 'S. S. Kusumawardani and S. A. I. Alfarozi, "Transformer encoder model for sequential prediction of student performance based on their log activities," <i>IEEE Access</i>, vol. 11, pp. 18960–18971, 2023, doi: 10.1109/ACCESS.2023.3246122.',
"liu2023-mcag": 'Y. Liu, Y. Hui, D. Hou, and X. Liu, "A novel student achievement prediction method based on deep learning and attention mechanism," <i>IEEE Access</i>, vol. 11, pp. 87245–87255, 2023, doi: 10.1109/ACCESS.2023.3305248.',
"vives2024-lstm": 'L. Vives et al., "Prediction of students\' academic performance in the programming fundamentals course using long short-term memory neural networks," <i>IEEE Access</i>, vol. 12, pp. 5882–5898, 2024, doi: 10.1109/ACCESS.2024.3350169.',
"alnasyan2025-dl": 'B. Alnasyan, M. Basheri, and M. O. Alassafi, "A comprehensive comparative analysis of deep learning models for student performance prediction in virtual learning environments: Leveraging the OULA dataset and advanced resampling techniques," <i>IEEE Access</i>, vol. 13, pp. 75953–75972, 2025, doi: 10.1109/ACCESS.2025.3564719.',
"prabowo2021-gpa": 'H. Prabowo, A. A. Hidayat, T. W. Cenggoro, R. Rahutomo, K. Purwandari, and B. Pardamean, "Aggregating time series and tabular data in deep learning model for university students\' GPA prediction," <i>IEEE Access</i>, vol. 9, pp. 87370–87377, 2021, doi: 10.1109/ACCESS.2021.3088152.',
"yousafzai2021-bilstm": 'B. K. Yousafzai et al., "Student-Performulator: Student academic performance using hybrid deep neural network," <i>Sustainability</i>, vol. 13, no. 17, Aug. 2021, Art. no. 9775, doi: 10.3390/su13179775.',
"duong2023-warning": 'H. Duong, L. T.-M. Tran, H. Q. To, and K. V. Nguyen, "Academic performance warning system based on data driven for higher education," <i>Neural Comput. Appl.</i>, vol. 35, no. 8, pp. 5819–5837, Mar. 2023, doi: 10.1007/s00521-022-07997-6.',
"miranda2024-interpretability": 'E. Miranda, M. Aryuni, M. I. Rahmawati, S. E. Hiererra, and A. V. D. Sano, "Machine learning\'s model-agnostic interpretability on the prediction of students\' academic performance in video-conference-assisted online learning during the COVID-19 pandemic," <i>Comput. Educ. Artif. Intell.</i>, vol. 7, 2024, Art. no. 100312, doi: 10.1016/j.caeai.2024.100312.',
# --- Nguồn chính thức HSA (grey literature, không có DOI) ---
"vnu-idt2026-spec": 'Institute for Digital Training and Testing, Vietnam National Univ., Hanoi, "Decision 01/QĐ-ĐTSKT on the format and detailed specification of the High-school Student Assessment (HSA) test 2026," Hanoi, Vietnam, Jan. 5, 2026. [CẦN URL + ngày truy cập]',
"vnu2026-scores": 'Vietnam National Univ., Hanoi, "Score distribution and percentiles of the High-school Student Assessment (HSA) 2026." [Online]. Available: https://vnu.edu.vn [CẦN URL đầy đủ + ngày truy cập]',
"vnu-hsa-institutions": 'Institute for Digital Training and Testing, Vietnam National Univ., Hanoi, "List of institutions using HSA results." [Online]. Available: https://hsa.edu.vn [CẦN URL đầy đủ + ngày truy cập]',
}


def main():
    s = open(DRAFT).read()
    # Bắt MỌI [token] trông giống citation key (chữ thường + gạch nối, không có
    # khoảng trắng). KHÔNG đòi 4 chữ số: key nguồn chính thức như
    # 'vnu-hsa-institutions' không có năm -> regex cũ bỏ sót âm thầm.
    cands = re.findall(r"\[([a-z][a-z0-9-]{3,})\]", s)
    order, seen = [], set()
    for k in cands:
        if k in BIB and k not in seen:
            seen.add(k); order.append(k)

    # Key trông giống citation nhưng KHÔNG có trong BIB -> dừng, không bịa
    unknown = sorted({k for k in cands if k not in BIB})
    if unknown:
        print("LỖI — key trích dẫn KHÔNG có trong thư mục đã verify:", file=sys.stderr)
        for m in unknown:
            print("   ", m, file=sys.stderr)
        print("Không sinh file. Thêm ref (đã verify) vào BIB hoặc sửa key.", file=sys.stderr)
        sys.exit(1)

    num = {k: i + 1 for i, k in enumerate(order)}
    for k, n in num.items():
        s = s.replace(f"[{k}]", f"[{n}]")

    refs = "\n".join(f"[{n}] {BIB[k]}\n" for k, n in num.items())
    s = re.sub(r"## REFERENCES\n\n\[TBD.*?\]", "## REFERENCES\n\n" + refs, s,
               flags=re.S)
    open(OUT, "w").write(s)

    unused = set(BIB) - set(order)
    print(f"Đã đánh số {len(order)} tài liệu -> {OUT}")
    if unused:
        print(f"({len(unused)} ref có trong BIB nhưng chưa trích dẫn: "
              f"{', '.join(sorted(unused))})")


if __name__ == "__main__":
    main()
