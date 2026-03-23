# =============================================================================
# FakeNews Multimodal System — Makefile
# Colab ve lokal ortam için tam iş akışı rehberi
# =============================================================================
#
# HIZLI BAŞLANGIÇ (Colab'da sırasıyla çalıştır):
#
#   1.  make pull          → GitHub'dan temiz çek (eski dosyaları siler, remote'a sıfırlar)
#   2.  make install       → pip bağımlılıklarını kur
#   3.  make setup-nltk    → NLTK veri paketlerini indir
#   4.  make link-data     → Drive klasörlerini data/ ve outputs/ 'a bağla
#   5.  make check-all     → GPU + veri + import hepsini doğrula
#   6.  make train         → Model 1'i sıfırdan eğit (45–90 dk)
#   7.  make show-results  → Eğitim metriklerini yazdır
#   8.  make predict TEXT="haber metni" → Tekil çıkarım
#
# UYARI: make pull yerel değişikliklerinizi siler.
#        data/ ve outputs/ symlink'leri korunur (Drive'a bağlıdır).
#
# =============================================================================

PYTHON      ?= python
REPO_URL    := https://github.com/fatih-yigit61/FakeNews-Multimodal-System.git
REPO_DIR    := FakeNews-Multimodal-System
# PROJECT_DIR: Makefile repo kökünde çalıştırılıyorsa REPO_DIR,
# zaten repo içindeysek "." olur. main.py varlığına bakarak karar verir.
PROJECT_DIR := $(shell if [ -f main.py ]; then echo "."; else echo "$(REPO_DIR)"; fi)

# Google Drive dizinleri — kendi Drive yapınıza göre ayarlayın
# DRIVE_DATA    : SemEval train-articles ve train-labels klasörlerinin bulunduğu dizin
# DRIVE_OUTPUTS : model checkpoint'lerinin kaydedileceği dizin (runtime'da kaybolmaz)
# WELFAKE_CSV   : WELFake_Dataset.csv dosyasının tam yolu (fake_head eğitimi için)
DRIVE_DATA    ?= /content/drive/MyDrive/Thesis_Results/semeval_data
DRIVE_OUTPUTS ?= /content/drive/MyDrive/Thesis_Results/outputs
WELFAKE_CSV   ?= /content/drive/MyDrive/Thesis_Results/WELFake_Dataset.csv

# Sabit yollar (değiştirmeyin)
DATA_DIR    := $(PROJECT_DIR)/data
OUTPUTS_DIR := $(PROJECT_DIR)/outputs
MODEL_OUT   := $(OUTPUTS_DIR)/model1
GNN_OUT     := $(OUTPUTS_DIR)/gnn_features

.PHONY: help pull install setup-nltk link-data check-gpu check-data \
        check-imports check-all train predict export-gnn show-results plots \
        test show-test-results clean clean-checkpoints clean-gnn full-pipeline \
        threshold-tuning style-ablation adversarial-test attention-analysis \
        error-analysis head-ablation analysis-all

# =============================================================================
# YARDIM
# =============================================================================

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║        FakeNews Multimodal System — Komut Rehberi            ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "── KURULUM (Bu sırayla çalıştır) ──────────────────────────────"
	@echo ""
	@echo "  make pull          GitHub'dan son kodu çek (veya ilk kez clone)"
	@echo "                     Dosya: .git/  Üretir: güncell. kaynak kodlar"
	@echo ""
	@echo "  make install       pip bağımlılıklarını kur"
	@echo "                     Dosya: requirements.txt  Üretir: kurulu paketler"
	@echo ""
	@echo "  make setup-nltk    NLTK punkt verisi indir (cümle tokenizer)"
	@echo "                     Gerekli: data_loader.py nltk.sent_tokenize kullanıyor"
	@echo ""
	@echo "  make link-data     Drive klasörlerini data/ ve outputs/ 'a bağla"
	@echo "                     DRIVE_DATA=$(DRIVE_DATA)"
	@echo "                     DRIVE_OUTPUTS=$(DRIVE_OUTPUTS)"
	@echo "                     Beklenen Drive yapısı:"
	@echo "                       DRIVE_DATA/"
	@echo "                         train-articles/          ← SemEval makaleler"
	@echo "                         train-labels-task2-technique-classification/"
	@echo "                                                  ← SemEval etiketler"
	@echo "                       WELFAKE_CSV                ← WELFake_Dataset.csv (fake_head)"
	@echo "                       DRIVE_OUTPUTS/             ← checkpoint kaydı"
	@echo ""
	@echo "── DOĞRULAMA ───────────────────────────────────────────────────"
	@echo ""
	@echo "  make check-gpu     GPU'nun aktif olup olmadığını kontrol et"
	@echo "  make check-data    Veri dizinlerinin varlığını ve içeriğini doğrula"
	@echo "  make check-imports Tüm Python modüllerinin import edildiğini test et"
	@echo "  make check-all     Üçünü birden çalıştır (önerilen)"
	@echo ""
	@echo "── EĞİTİM ──────────────────────────────────────────────────────"
	@echo ""
	@echo "  make train         Model 1'i sıfırdan eğit + GNN özellik ihracatı"
	@echo "                     Giriş  : data/train-articles/          (SemEval → manip_head)"
	@echo "                              data/train-labels-task2-*/"
	@echo "                              data/WELFake_Dataset.csv      (WELFake → fake_head)"
	@echo "                     Çıktı  : outputs/model1/best_model.pt"
	@echo "                              outputs/model1/style_scaler.npz"
	@echo "                              outputs/model1/tokenizer/"
	@echo "                              outputs/model1/training_history.json"
	@echo "                              outputs/gnn_features/feature_matrix.pt"
	@echo "                     Süre   : L4/A100 GPU'da ~45–90 dakika"
	@echo ""
	@echo "── ÇIKARIM ─────────────────────────────────────────────────────"
	@echo ""
	@echo "  make test          Eğitim bittikten sonra held-out test değerlendirmesi"
	@echo "                     Gerekli: outputs/model1/best_model.pt"
	@echo "                     Test setleri:"
	@echo "                       semeval_test    : SemEval dev-articles/ veya %15 held-out"
	@echo "                       welfake_test    : WELFake'in %10 held-out dilimi"
	@echo "                       tweet_eval_test : tweet_eval HF native test split"
	@echo "                       isot_cross_domain : ISOT (eğitimde yok → cross-domain)"
	@echo "                     Çıktı  : outputs/model1/test_results.json"
	@echo ""
	@echo "  make predict TEXT=\"metin\"   Tek metin için tüm görev çıktıları"
	@echo "                     Gerekli: outputs/model1/best_model.pt (eğitim sonrası)"
	@echo "                     Çıktı  : fake_score, manipulation_score, sentiment"
	@echo ""
	@echo "  make export-gnn    UPFD haberleri için 128-d GNN özellik vektörleri"
	@echo "                     Gerekli: outputs/model1/best_model.pt"
	@echo "                              data/upfd/news_content.json"
	@echo "                     Çıktı  : outputs/gnn_features/{id}.pt"
	@echo "                              outputs/gnn_features/feature_matrix.pt"
	@echo "                              outputs/gnn_features/index.json"
	@echo ""
	@echo "── SONUÇLAR ────────────────────────────────────────────────────"
	@echo ""
	@echo "  make show-results  Eğitim geçmişini (training_history.json) yazdır"
	@echo ""
	@echo "── TEMİZLİK ────────────────────────────────────────────────────"
	@echo ""
	@echo "  make clean-checkpoints   Sadece model ağırlıklarını sil"
	@echo "  make clean-gnn           GNN özellik dosyalarını sil"
	@echo "  make clean               Her ikisini de temizle"
	@echo ""
	@echo "── TAM PIPELINE ────────────────────────────────────────────────"
	@echo ""
	@echo "  make full-pipeline  pull → install → setup-nltk → link-data"
	@echo "                      → check-all → train hepsini sırasıyla çalıştırır"
	@echo ""
	@echo "── DEĞİŞKENLER ─────────────────────────────────────────────────"
	@echo ""
	@echo "  make plots         training_history.json'dan grafikleri yeniden üret"
	@echo "                     Çıktı: outputs/model1/plots/*.png"
	@echo ""
	@echo "  make show-test-results  test_results.json'u okunabilir tablo olarak yazdır"
	@echo ""
	@echo "── DEĞİŞKENLER ─────────────────────────────────────────────"
	@echo ""
	@echo "  DRIVE_DATA     SemEval veri dizini     (varsayılan: $(DRIVE_DATA))"
	@echo "  DRIVE_OUTPUTS  Checkpoint kayıt dizini (varsayılan: $(DRIVE_OUTPUTS))"
	@echo "  WELFAKE_CSV    WELFake CSV tam yolu    (varsayılan: $(WELFAKE_CSV))"
	@echo "  PYTHON         Python yorumlayıcısı    (varsayılan: python)"
	@echo "  TEXT           make predict için metin"
	@echo ""
	@echo "  Örnek:"
	@echo "    make link-data \\"
	@echo "      DRIVE_DATA=/content/drive/MyDrive/Thesis_Results/semeval_data \\"
	@echo "      DRIVE_OUTPUTS=/content/drive/MyDrive/Thesis_Results/outputs \\"
	@echo "      WELFAKE_CSV=/content/drive/MyDrive/Thesis_Results/WELFake_Dataset.csv"
	@echo ""

# =============================================================================
# KURULUM
# =============================================================================

# GitHub'dan en güncel kodu çeker.
# Repo varsa: remote'a sıfırlar (reset --hard) + izlenmeyen dosyaları siler (clean -fd).
# Bu sayede eski yapıdan kalan dosyalar veya klasörler tamamen temizlenir.
# Repo yoksa: clone yapar.
#
# NOT: Yerel yaptığın değişiklikler (commit edilmemiş) bu komutla silinir.
#      Sadece Drive'a bağlı data/ ve outputs/ symlink'leri etkilenmez.
pull:
	@echo ">>> GitHub'dan temiz çekme: $(REPO_URL)"
	@if [ -d "$(REPO_DIR)/.git" ]; then \
		echo "    Repo mevcut — remote'a sıfırlanıyor..."; \
		cd $(REPO_DIR) && git fetch origin main && \
		git reset --hard origin/main && \
		git clean -fd; \
	else \
		echo "    Repo yok, klonlanıyor..."; \
		git clone $(REPO_URL); \
	fi
	@echo "    Son commit:"
	@cd $(REPO_DIR) && git log --oneline -3
	@echo "    Son commit:"
	@cd $(REPO_DIR) && git log --oneline -3

# requirements.txt içindeki tüm bağımlılıkları kurar.
# Colab'da torch/transformers zaten vardır; sadece eksikler yüklenir.
install:
	@echo ">>> Bağımlılıklar kuruluyor..."
	$(PYTHON) -m pip install -q -U pip
	$(PYTHON) -m pip install -q -r $(PROJECT_DIR)/requirements.txt
	@echo "    Kurulum tamamlandı."

# data_loader.py, nltk.sent_tokenize() kullanır.
# Bu komut olmadan parse() çağrısında LookupError alırsın.
setup-nltk:
	@echo ">>> NLTK veri paketleri indiriliyor..."
	$(PYTHON) -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); print('    punkt OK')"

# Drive'daki veri klasörlerini data/ ve outputs/ altına sembolik link ile bağlar.
# Python tabanlı auto-detect: "semeval_data" veya "semeval_data " (trailing space)
# fark etmez, otomatik bulur. Manuel path vermeye gerek yok.
link-data:
	@echo ">>> Drive klasörleri bağlanıyor (auto-detect)..."
	@mkdir -p "$(DRIVE_OUTPUTS)/model1" "$(DRIVE_OUTPUTS)/gnn_features"
	@mkdir -p $(DATA_DIR)
	@rm -f $(PROJECT_DIR)/outputs
	ln -sfn "$(DRIVE_OUTPUTS)" $(PROJECT_DIR)/outputs
	$(PYTHON) -c "\
import os, glob; \
drive_root = '/content/drive/MyDrive/Thesis_Results'; \
data_dir = '$(DATA_DIR)'; \
# --- SemEval: trailing space auto-detect --- \
candidates = sorted(glob.glob(os.path.join(drive_root, 'semeval_data*'))); \
semeval = None; \
for c in candidates: \
    if os.path.isdir(c) and os.path.isdir(os.path.join(c, 'train-articles')): \
        semeval = c; break; \
if semeval: \
    print(f'    SemEval bulundu: {repr(semeval)}'); \
    for name in ['train-articles', 'train-labels-task2-technique-classification']: \
        dst = os.path.join(data_dir, name); \
        os.path.lexists(dst) and os.remove(dst); \
        os.symlink(os.path.join(semeval, name), dst); \
        print(f'    linked: {name}'); \
else: \
    print('    [HATA] semeval_data* klasoru bulunamadi: ' + drive_root); \
# --- UPFD --- \
upfd_src = os.path.join(drive_root, 'upfd_data'); \
upfd_dst = os.path.join(data_dir, 'upfd'); \
os.path.lexists(upfd_dst) and os.remove(upfd_dst); \
if os.path.isdir(upfd_src): \
    os.symlink(upfd_src, upfd_dst); \
    print('    linked: upfd'); \
else: \
    print('    [uyari] upfd_data bulunamadi (opsiyonel)'); \
# --- WELFake --- \
welfake_src = os.path.join(drive_root, 'WELFake_Dataset.csv'); \
welfake_dst = os.path.join(data_dir, 'WELFake_Dataset.csv'); \
os.path.lexists(welfake_dst) and os.remove(welfake_dst); \
if os.path.isfile(welfake_src): \
    os.symlink(welfake_src, welfake_dst); \
    print('    linked: WELFake_Dataset.csv'); \
else: \
    print('    [HATA] WELFake_Dataset.csv bulunamadi'); \
# --- Dogrulama --- \
print(); \
import subprocess; \
r = subprocess.run(['ls', '-la', data_dir], capture_output=True, text=True); \
print(r.stdout); \
"
	@echo "    Outputs → $(DRIVE_OUTPUTS)"

# =============================================================================
# DOĞRULAMA
# =============================================================================

# GPU ve CUDA kontrolü.
# Eğer GPU yoksa: Runtime > Change runtime type > T4/L4/A100 GPU seçin.
check-gpu:
	@echo ">>> GPU kontrolü..."
	$(PYTHON) -c "\
import torch; \
gpu = torch.cuda.is_available(); \
print('    CUDA:', gpu); \
print('    GPU :', torch.cuda.get_device_name(0) if gpu else 'YOK — Runtime menüsünden GPU seçin'); \
print('    Bellek:', f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB' if gpu else '-')"

# Veri dizinlerinin varlığını ve içeriğini kontrol eder.
check-data:
	@echo ">>> Veri kontrolü..."
	@echo "    train-articles:"
	@ls $(DATA_DIR)/train-articles/*.txt 2>/dev/null | wc -l | xargs echo "      .txt dosya sayısı:"
	@echo "    train-labels:"
	@ls $(DATA_DIR)/train-labels-task2-technique-classification/ 2>/dev/null | wc -l | xargs echo "      etiket dosya sayısı:"
	@echo "    UPFD (opsiyonel):"
	@test -f $(DATA_DIR)/upfd/news_content.json && echo "      news_content.json OK" || echo "      news_content.json YOK (GNN export atlanır)"
	@echo "    Outputs dizini:"
	@test -d $(PROJECT_DIR)/outputs/model1 && echo "      outputs/model1 OK" || echo "      outputs/model1 YOK"

# Tüm Python modüllerinin import edildiğini test eder.
check-imports:
	@echo ">>> Python import kontrolü..."
	@cd $(PROJECT_DIR) && $(PYTHON) -c "\
from configs.config import TrainerConfig, ENHANCED_DIM, MANIP_EMBED_DIM; \
from src.models.text_transformer import OptimizedMultiTaskModel; \
from src.training.text_trainer import Model1ExpertTrainer; \
from src.training.loss import MultiTaskLoss; \
from src.preprocessing.data_loader import SemEvalParser, PropagandaDataset, SimpleNewsDataset; \
from src.features.stylometry import StylometricExtractor, StyleScaler; \
from src.features.gnn_exporter import GNNFeatureExporter; \
print('    Tüm importlar OK'); \
print(f'    ENHANCED_DIM={ENHANCED_DIM}  MANIP_EMBED_DIM={MANIP_EMBED_DIM}')"

# Üç kontrolü birden çalıştırır. Eğitimden önce bunu çalıştırın.
check-all: check-gpu check-data check-imports
	@echo ""
	@echo ">>> Tüm kontroller tamamlandı. Eğitime hazır."

# =============================================================================
# EĞİTİM
# =============================================================================

# Model 1'i sıfırdan eğitir, ardından GNN özellik ihracatını çalıştırır.
# UYARI: Bu komut eski checkpoint'in üzerine yazar.
#        Eski modeli korumak istiyorsanız önce DRIVE_OUTPUTS/model1/ klasörünü yedekleyin.
train:
	@echo ">>> Model 1 eğitimi başlıyor..."
	@echo "    Giriş : $(DATA_DIR)/train-articles/"
	@echo "    Çıktı : $(DRIVE_OUTPUTS)/model1/best_model.pt"
	@echo "    Süre  : L4/A100 GPU'da ~45–90 dakika"
	@echo ""
	cd $(PROJECT_DIR) && $(PYTHON) main.py --train
	@echo ""
	@echo ">>> Eğitim tamamlandı. Sonuçları görmek için: make show-results"

# =============================================================================
# ÇIKARIM
# =============================================================================

# Eğitilmiş modeli yükler ve tek bir metin için 4 görevin sonuçlarını verir.
# Kullanım: make predict TEXT="analiz edilecek haber metni"
predict:
	@if [ -z "$(TEXT)" ]; then \
		echo ">>> Örnek metin ile çıkarım yapılıyor..."; \
		cd $(PROJECT_DIR) && $(PYTHON) main.py --predict; \
	else \
		echo ">>> Çıkarım: $(TEXT)"; \
		cd $(PROJECT_DIR) && $(PYTHON) main.py --predict --text "$(TEXT)"; \
	fi

# UPFD haber corpus'u üzerinde 128-d manipulation embedding vektörlerini üretir.
export-gnn:
	@echo ">>> GNN özellik ihracatı başlıyor..."
	@test -f $(PROJECT_DIR)/outputs/model1/best_model.pt || (echo "    HATA: best_model.pt bulunamadı. Önce 'make train' çalıştırın." && exit 1)
	@test -f $(DATA_DIR)/upfd/news_content.json || (echo "    HATA: news_content.json bulunamadı. Drive'da upfd/ klasörü gerekli." && exit 1)
	cd $(PROJECT_DIR) && $(PYTHON) -c "\
from configs.config import TrainerConfig; \
from src.training.text_trainer import Model1ExpertTrainer; \
from src.features.gnn_exporter import GNNFeatureExporter; \
trainer = Model1ExpertTrainer(TrainerConfig()); \
trainer.build_model(); \
trainer.load_best_model(); \
exporter = GNNFeatureExporter(trainer); \
path = exporter.export(overwrite=False); \
print('İhracat tamamlandı:', path)"

# =============================================================================
# TEST DEĞERLENDİRMESİ (eğitim bittikten sonra çalıştır)
# =============================================================================

# Held-out test setleri üzerinde evaluate_all() çalıştırır.
# Eşik (manipulation threshold) val seti üzerinde kalibre edilir, test setinde kullanılır.
# ISOT cross-domain testi: model WELFake'te eğitildi, hiç görmediği ISOT'ta test ediliyor.
test:
	@echo ">>> Held-out test değerlendirmesi başlıyor..."
	@test -f $(PROJECT_DIR)/outputs/model1/best_model.pt || \
		(echo "    HATA: best_model.pt bulunamadı. Önce 'make train' çalıştırın." && exit 1)
	cd $(PROJECT_DIR) && $(PYTHON) main.py --test
	@echo ""
	@echo ">>> Sonuçlar: outputs/model1/test_results.json"
	@echo "    Cross-domain not: isot_cross_domain skoru, modelin WELFake dışına genelleyebildiğini gösterir."
	@echo "    WELFake ve ISOT arasında metin çakışması varsa sonucu yorumlarken dikkatli ol."

# Held-out test sonuçlarını tam metrik tablosu olarak gösterir (test_results.json varsa)
show-test-results:
	@test -f $(PROJECT_DIR)/outputs/model1/test_results.json || \
		(echo "    test_results.json bulunamadı. Önce 'make test' çalıştırın." && exit 1)
	@cd $(PROJECT_DIR) && $(PYTHON) -c "\
import json; \
from src.training.text_trainer import Model1ExpertTrainer; \
r = json.load(open('outputs/model1/test_results.json')); \
Model1ExpertTrainer._print_test_results(r)"

# =============================================================================
# SONUÇLAR
# =============================================================================

# training_history.json'dan grafikleri yeniden üretir (eğitim bitmişse, grafik yoksa)
plots:
	@echo ">>> Eğitim grafikleri oluşturuluyor..."
	@test -f $(PROJECT_DIR)/outputs/model1/training_history.json || \
		(echo "    HATA: training_history.json bulunamadı. Önce 'make train' çalıştırın." && exit 1)
	@cd $(PROJECT_DIR) && $(PYTHON) -c "\
import json; \
from src.training.text_trainer import Model1ExpertTrainer; \
h = json.load(open('outputs/model1/training_history.json')); \
Model1ExpertTrainer._save_training_plots(h); \
print('Grafikler: outputs/model1/plots/')"

# Eğitim geçmişini (training_history.json) okunabilir biçimde yazdırır.
show-results:
	@echo ">>> Eğitim sonuçları:"
	@test -f $(PROJECT_DIR)/outputs/model1/training_history.json || \
		(echo "    training_history.json bulunamadı. Eğitim yapılmamış olabilir." && exit 1)
	@cd $(PROJECT_DIR) && $(PYTHON) -c "\
import json; \
h = json.load(open('outputs/model1/training_history.json')); \
print(f'  {\"Epoch\":<6} {\"TrainLoss\":<12} {\"ManipF1\":<10} {\"FakeAcc\":<10} {\"SentAcc\":<10} {\"Composite\":<10}'); \
print('  ' + '-'*64); \
[print(f'  {e[\"epoch\"]:<6} {e[\"train_loss\"]:<12.4f} {e[\"manipulation_f1\"]:<10.4f} {e.get(\"fake_acc\",0):<10.4f} {e.get(\"sentiment_acc\",0):<10.4f} {e[\"composite_score\"]:<10.4f}') for e in h]; \
best = max(h, key=lambda x: x['composite_score']); \
sent_ok = best.get('sentiment_acc', 0) >= 0.75; \
manip_ok = best['manipulation_f1'] >= 0.65; \
fake_ok = best.get('fake_acc', 0) >= 0.85; \
print(); \
print(f'  En iyi epoch   : {best[\"epoch\"]}'); \
print(f'  Composite skor : {best[\"composite_score\"]:.4f}'); \
print(); \
print(f'  ── Hedef Kontrolü ─────────────────────────────────────────'); \
print(f'  Fake/Real  >= 85%  : {best.get(\"fake_acc\",0):.4f}  {\"✅\" if fake_ok  else \"❌\"}'); \
print(f'  SentAcc    >= 75%  : {best.get(\"sentiment_acc\",0):.4f}  {\"✅\" if sent_ok  else \"❌\"}'); \
print(f'  ManipF1    >= 0.65 : {best[\"manipulation_f1\"]:.4f}  {\"✅\" if manip_ok else \"❌\"}'); \
print(); \
import os; \
plots = 'outputs/model1/plots'; \
os.path.isdir(plots) and print(f'  Grafikler: {plots}/ (training_curves.png, manipulation.png, fake.png, sentiment.png, composite.png)')"

# =============================================================================
# TEMİZLİK
# =============================================================================

# Sadece model ağırlıklarını siler (yeniden eğitmek için)
clean-checkpoints:
	@echo ">>> Checkpoint temizleniyor..."
	rm -f $(PROJECT_DIR)/outputs/model1/best_model.pt
	rm -f $(PROJECT_DIR)/outputs/model1/style_scaler.npz
	rm -rf $(PROJECT_DIR)/outputs/model1/tokenizer/
	@echo "    Silindi: best_model.pt, style_scaler.npz, tokenizer/"

# GNN özellik dosyalarını siler (yeniden ihracat için)
clean-gnn:
	@echo ">>> GNN özellikleri temizleniyor..."
	rm -f $(PROJECT_DIR)/outputs/gnn_features/*.pt
	rm -f $(PROJECT_DIR)/outputs/gnn_features/index.json
	@echo "    Silindi: gnn_features/*.pt ve index.json"

# İkisini birden temizler
clean: clean-checkpoints clean-gnn
	@echo ">>> Tüm çıktılar temizlendi."

# =============================================================================
# FAZ A: THRESHOLD TUNING (eğitim sonrası, 0 token maliyeti)
# =============================================================================

# Manipulation threshold sweep (0.20-0.80) + sentiment temperature scaling
# Grafik çıktıları: threshold_vs_F1, reliability diagrams
threshold-tuning:
	@echo ">>> Post-hoc threshold & calibration tuning..."
	@test -f $(PROJECT_DIR)/outputs/model1/best_model.pt || \
		(echo "    HATA: best_model.pt bulunamadı. Önce 'make train' çalıştırın." && exit 1)
	cd $(PROJECT_DIR) && $(PYTHON) scripts/threshold_tuning.py
	@echo "    Grafikler: outputs/model1/plots/manipulation_threshold_calibration.png"
	@echo "               outputs/model1/plots/reliability_*.png"

# =============================================================================
# FAZ B: MODEL VALIDATION (0 token maliyeti, retraining yok)
# =============================================================================

# B1: Style feature ablation — style_feats=0 ile test
style-ablation:
	@echo ">>> Style feature ablation study..."
	@test -f $(PROJECT_DIR)/outputs/model1/best_model.pt || \
		(echo "    HATA: best_model.pt bulunamadı." && exit 1)
	cd $(PROJECT_DIR) && $(PYTHON) scripts/style_ablation.py

# B2: Adversarial test — 15 real/fake cümle çifti
adversarial-test:
	@echo ">>> Adversarial robustness test..."
	@test -f $(PROJECT_DIR)/outputs/model1/best_model.pt || \
		(echo "    HATA: best_model.pt bulunamadı." && exit 1)
	cd $(PROJECT_DIR) && $(PYTHON) scripts/adversarial_test.py

# B3: Attention visualization — Captum Integrated Gradients + attention heatmaps
attention-analysis:
	@echo ">>> Attention & interpretability analysis..."
	@test -f $(PROJECT_DIR)/outputs/model1/best_model.pt || \
		(echo "    HATA: best_model.pt bulunamadı." && exit 1)
	$(PYTHON) -m pip install -q captum 2>/dev/null || true
	cd $(PROJECT_DIR) && $(PYTHON) scripts/attention_analysis.py

# B4: Error analysis — yanlış tahminlerin pattern analizi
error-analysis:
	@echo ">>> Error analysis..."
	@test -f $(PROJECT_DIR)/outputs/model1/best_model.pt || \
		(echo "    HATA: best_model.pt bulunamadı." && exit 1)
	cd $(PROJECT_DIR) && $(PYTHON) scripts/error_analysis.py

# Tüm Faz B analizlerini sırasıyla çalıştır
analysis-all: threshold-tuning style-ablation adversarial-test attention-analysis error-analysis
	@echo ""
	@echo "╔══════════════════════════════════════════════╗"
	@echo "║  Tüm analizler tamamlandı!                   ║"
	@echo "║  Grafikler: outputs/model1/plots/            ║"
	@echo "╚══════════════════════════════════════════════╝"

# =============================================================================
# FAZ C: HEAD ABLATION (extra Colab session gerektirir)
# =============================================================================

# Only-Fake-Head vs Full Model karşılaştırma (retraining gerekir)
head-ablation:
	@echo ">>> Head ablation study (Only Fake vs Full Model)..."
	@echo "    UYARI: Bu komut yeni bir eğitim başlatır (~1 saat)"
	cd $(PROJECT_DIR) && $(PYTHON) scripts/head_ablation.py

# =============================================================================
# TAM PIPELINE (tek komutla her şeyi çalıştır)
# =============================================================================

# pull → install → setup-nltk → link-data → check-all → train
# Colab'da sıfırdan başlarken tek bir komutla her şeyi çalıştırır.
full-pipeline: pull install setup-nltk link-data check-all train
	@echo ""
	@echo "╔══════════════════════════════════════════════╗"
	@echo "║  Full pipeline tamamlandı!                   ║"
	@echo "║  Sonuçlar için: make show-results            ║"
	@echo "║  Çıkarım için:  make predict TEXT=\"...\"      ║"
	@echo "╚══════════════════════════════════════════════╝"

# train → threshold-tuning → tüm analizler (Faz A + B tek seferde)
full-analysis: train threshold-tuning analysis-all
	@echo ""
	@echo "╔══════════════════════════════════════════════╗"
	@echo "║  Eğitim + Tüm analizler tamamlandı!         ║"
	@echo "║  Grafikler: outputs/model1/plots/            ║"
	@echo "╚══════════════════════════════════════════════╝"