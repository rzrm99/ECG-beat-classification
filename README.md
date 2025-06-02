# ECG-beat-classification
## 🧠 What This ECG Model Does

This project implements a deep learning model that performs **ECG beat classification** using **1D Convolutional Neural Networks (CNNs)**. It is trained on the **MIT-BIH Arrhythmia Database**.

---

### ✅ Prediction Task

For each input heartbeat (a segment of 187 ECG signal values), the model predicts its class from the following categories:

| Label | Description                              |
|-------|------------------------------------------|
| 0     | Normal beat                              |
| 1     | Supraventricular ectopic beat (S)        |
| 2     | Ventricular ectopic beat (V)             |
| 3     | Fusion beat                              |
| 4     | Unknown or unclassifiable beat           |

---

### 🚨 Detection Capability

By predicting the beat class, the model can detect:

- **Irregular heart rhythms (arrhythmias)**
- **Ectopic beats** (abnormal origin beats)
- **Potential signs of cardiac distress or failure**

This makes it suitable for powering **real-time ECG monitoring systems** that trigger alerts when abnormal heart activity is detected.

---

### 💡 Use Cases

- Wearable health monitors
- Hospital ECG alert systems
- Real-time heart rhythm classification
- Personal health analytics apps


  ## 📁 Dataset

This project uses the **MIT-BIH Arrhythmia-derived ECG Classification Dataset**, available publicly on Mendeley Data.

- **Name**: ECG Classification Dataset
- **Source**: [Mendeley Data - txhsxnsm6d](https://data.mendeley.com/datasets/txhsxnsm6d/1)
- **Format**: CSV files (`mitbih_train.csv`, `mitbih_test.csv`)
- **Description**: Preprocessed ECG heartbeat segments labeled by type (normal, ectopic, etc.), suitable for supervised learning.

Make sure to download both the training and test sets before running the model.



## ⚠️ Disclaimer

This project is provided for **educational and research purposes only**.

- **Not for clinical use**: The model has not been tested, verified, or approved for use in real-life medical or diagnostic situations.
- **No medical guarantees**: It is not intended to diagnose, treat, or prevent any health conditions, and should not be used as a replacement for professional medical advice.
- **Use at your own risk**: The author does not accept any responsibility or liability for potential consequences resulting from the use or misuse of this software in any real-world application.

Always consult with a licensed healthcare professional for any medical concerns or decisions.


## 📄 License

This project is licensed under the [MIT License](LICENSE).

