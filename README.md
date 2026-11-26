# NonDelusional_ML

# 💳 Makine Öğrenmesi Algoritmalarıyla Kredi Risk Modellemesi

> *"Bir krediyi onaylamak bir güven meselesidir. Biz de bu güveni verinin diliyle ölçmeye çalışıyoruz."*

Bu proje, bir kredi başvurusunun **iyi** ya da **riskli** olup olmadığını makine öğrenmesi yöntemleriyle tahmin etmeyi amaçlıyor. Klasik **German Credit** veri seti üzerinde farklı algoritmaları adil koşullarda karşılaştırıyor, en iyi modeli seçiyor ve bu modelin *neden* öyle karar verdiğini açıklamaya çalışıyoruz.

Bu çalışma bir Makine Öğrenmesi dersi kapsamında hazırlandı. Ama hedefimiz yalnızca bir ödev teslim etmek değil, gerçek bir bankanın karşılaşacağı soruları ciddiye alan bir model kurmak.

---

## 📌 İçindekiler

- [Projenin Amacı](#-projenin-amacı)
- [Veri Seti](#-veri-seti)
- [Yöntem](#-yöntem)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Değerlendirme Yaklaşımı](#-değerlendirme-yaklaşımı)
- [Sonuçlar](#-sonuçlar)
- [Kısıtlar ve Etik](#-kısıtlar-ve-etik)
- [Yol Haritası](#-yol-haritası)
- [Kaynakça](#-kaynakça)

---

## 🎯 Projenin Amacı

Bankalar her gün yüzlerce kredi başvurusunu değerlendiriyor. Riskli bir müşteriye kredi vermek doğrudan para kaybı demek. Güvenilir bir müşteriyi reddetmek ise hem kaçırılmış bir fırsat hem de o kişi için haksızlık.

Bu projede şu sorulara cevap arıyoruz:

1. Başvuru sahibinin bilgilerinden **temerrüt riskini** ne kadar iyi tahmin edebiliriz?
2. Hangi makine öğrenmesi algoritması bu problemde daha başarılı?
3. Riski en çok **hangi faktörler** belirliyor?
4. Hataların maliyeti eşit değilken **en doğru karar eşiği** nedir?

---

## 📂 Veri Seti

**Statlog (German Credit Data)**: Prof. Dr. Hans Hofmann, Hamburg Üniversitesi. [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) üzerinden erişilebilir.

| Özellik | Değer |
|---|---|
| Gözlem sayısı | 1.000 başvuru |
| Değişken sayısı | 20 (7 sayısal, 13 kategorik) |
| Hedef değişken | `1 = İyi`, `2 = Kötü` (projede `0 / 1` olarak yeniden kodlanır) |
| Sınıf dağılımı | %70 iyi, %30 kötü |
| Eksik değer | Yok |

**Örnek değişkenler:** mevcut hesap durumu, kredi süresi (ay), kredi geçmişi, kredinin amacı, kredi tutarı, birikim durumu, istihdam süresi, yaş, konut durumu, meslek...

### ⚖️ Maliyet Matrisi

Bu veri setini özel kılan şey, beraberinde bir **maliyet matrisiyle** gelmesi:

| | Tahmin: İyi | Tahmin: Kötü |
|---|---|---|
| **Gerçek: İyi** | 0 | 1 |
| **Gerçek: Kötü** | **5** | 0 |

Yani kötü bir müşteriyi "iyi" diye sınıflandırmak, iyi bir müşteriyi reddetmekten **5 kat daha maliyetli**. Bu yüzden modelleri sadece doğrulukla değil, bu maliyete göre de değerlendiriyoruz.

---

## 🧠 Yöntem

```
Ham Veri → EDA → Ön İşleme → Özellik Mühendisliği → Modelleme → Değerlendirme → Açıklanabilirlik
```

**1. Keşifsel Veri Analizi:** Dağılımlar, değişkenlere göre temerrüt oranları ve korelasyonlar.

**2. Ön İşleme:** Kategorik değişkenlerin kodlanması (One-Hot / WoE), sayısal değişkenlerin ölçeklenmesi. Bütün adımlar veri sızıntısını önlemek için `scikit-learn Pipeline` içinde yapılır.

**3. Özellik Mühendisliği:** Weight of Evidence (WoE) ve Information Value (IV) analizi. Ayrıca *aylık taksit yükü* gibi türetilmiş değişkenler.

**4. Sınıf Dengesizliği:** `class_weight`, SMOTE ve undersampling yaklaşımlarının karşılaştırılması.

**5. Modeller:**

| Model | Neden? |
|---|---|
| Lojistik Regresyon | Temel model, yorumlanabilir, sektör standardı |
| Karar Ağacı | Görselleştirilebilir karar kuralları |
| Random Forest | Güçlü ve kararlı bir topluluk modeli |
| XGBoost / LightGBM | Tablo verilerinde genellikle en iyi performans |
| SVM, KNN | Farklı yaklaşımlarla karşılaştırma |

**6. Optimizasyon:** 5 katlı Stratified K-Fold çapraz doğrulama ile hiperparametre araması.

**7. Açıklanabilirlik:** SHAP değerleriyle hem genel hem de tek bir başvuru düzeyinde karar açıklaması.

---

## 🗂️ Proje Yapısı

```
cred/
├── data/
│   ├── raw/                  # Orijinal German Credit verisi
│   └── processed/            # Temizlenmiş ve dönüştürülmüş veri
├── notebooks/
│   ├── 01_eda.ipynb          # Keşifsel veri analizi
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb     # Model eğitimi ve karşılaştırma
│   └── 04_explainability.ipynb
├── src/                      # Yeniden kullanılabilir yardımcı fonksiyonlar
├── reports/
│   └── figures/              # Grafikler ve görseller
├── requirements.txt
└── README.md
```

---

## ⚙️ Kurulum

```bash
# Depoyu klonla
git clone <repo-adresi>
cd cred

# Sanal ortam oluştur ve etkinleştir
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

**Gereksinimler:** Python 3.10+

---

## 🚀 Kullanım

Notebook'ları sırayla çalıştırman yeterli:

```bash
jupyter lab
```

1. `01_eda.ipynb` ile veriyi tanı
2. `02_preprocessing.ipynb` ile veriyi modele hazırla
3. `03_modeling.ipynb` ile modelleri eğit ve karşılaştır
4. `04_explainability.ipynb` ile kararların arkasındaki nedenleri incele

Sonuçların tekrar üretilebilir olması için tüm rastgele işlemlerde `random_state=42` kullanılıyor.

---

## 📏 Değerlendirme Yaklaşımı

Veri dengesiz olduğu için **doğruluk (accuracy) tek başına yanıltıcı**. Her şeye "iyi" diyen bir model bile %70 doğruluk alır. Bu yüzden şu metrikleri kullanıyoruz:

- **ROC-AUC** ve **Gini katsayısı** (`Gini = 2 × AUC − 1`)
- **KS istatistiği**
- **Precision / Recall / F1**. Özellikle kötü sınıf için **Recall** önemli.
- **Toplam maliyet**: yukarıdaki maliyet matrisine göre hesaplanır
- **Optimum karar eşiği**: varsayılan 0.5 yerine toplam maliyeti en aza indiren eşik

---

## 📈 Sonuçlar

> 🚧 *Bu bölüm deneyler tamamlandıkça güncellenecek.*

| Model | ROC-AUC | Gini | Recall (Kötü) | Toplam Maliyet |
|---|---|---|---|---|
| Lojistik Regresyon | – | – | – | – |
| Karar Ağacı | – | – | – | – |
| Random Forest | – | – | – | – |
| XGBoost | – | – | – | – |
| LightGBM | – | – | – | – |

---

## 🤝 Kısıtlar ve Etik

Bu modeli kurarken teknik başarının yanında sorumluluğu da göz önünde tutmaya çalışıyoruz:

- **Veri sınırlı ve eski:** 1.000 gözlem ve 1990'ların Almanya'sı. Sonuçlar bugünün koşullarına doğrudan genellenemez.
- **Adalet (fairness):** Veride yaş, cinsiyet/medeni durum ve yabancı işçi olma gibi hassas değişkenler var. Modelin bu gruplara karşı sistematik bir önyargı üretip üretmediğini ayrıca inceliyoruz.
- **Açıklanabilirlik:** Bir kişinin kredisi reddediliyorsa bunun *nedenini* söyleyebilmeliyiz. SHAP analizinin bu kadar önemli olmasının sebebi bu.

---

## 🗺️ Yol Haritası

- [ ] Ortam kurulumu ve veri setinin indirilmesi
- [ ] Keşifsel veri analizi
- [ ] Ön işleme pipeline'ı
- [ ] WoE / IV analizi
- [ ] Temel modeller (LR, Karar Ağacı)
- [ ] Topluluk modelleri (RF, XGBoost, LightGBM)
- [ ] Dengesizlik yöntemlerinin karşılaştırılması
- [ ] Hiperparametre optimizasyonu ve eşik ayarı
- [ ] SHAP ile açıklanabilirlik
- [ ] Adalet analizi
- [ ] Rapor ve sunum

---

## 📚 Kaynakça

- Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- Siddiqi, N. (2017). *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards* (2nd ed.). Wiley.

---

<p align="center">
  <i>Hazırlayan: [Adın Soyadın] · [Ders Adı] · 2026</i><br>
  <sub>Sorular, öneriler ya da sadece bir merhaba için iletişime geçmekten çekinme. 🌱</sub>
</p>
