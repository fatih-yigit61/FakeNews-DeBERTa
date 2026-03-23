# FakeNews-Multimodal-System

Bu repo, tezindeki Model 1 (metin tabanlı fake news detection) kodunun modüler Python proje yapısını ve Colab entegrasyonunu içerir.

## Colab Üzerinden Çalıştırma

1. **Google Drive'ı mount et**

```python
from google.colab import drive
drive.mount("/content/drive")
```

2. **Kodu GitHub'dan çek**

```bash
%cd /content
!git clone https://github.com/<kullanici>/<repo-adi>.git
%cd <repo-adi>
```

3. **Bağımlılıkları kur ve Drive'daki verileri bağla**

```bash
!make install
!make link-data
```

4. **Model 1 eğitim (tam pipeline)**

```bash
!make train
```

5. **Örnek bir haber metni üzerinde tahmin**

```bash
!make predict TEXT="Bu haberin sahte olup olmadığını test etmek istiyorum."
```

Varsayılan olarak Makefile, Drive yolunu `/content/drive/MyDrive/Thesis_Results` olarak kabul eder. Farklı bir klasör kullanıyorsan Colab hücresinde şu şekilde override edebilirsin:

```bash
!DRIVE_ROOT="/content/drive/MyDrive/Baska_Klasor" make link-data
```

