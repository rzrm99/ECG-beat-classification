# ECG-beat-classification
🧠 What This ECG Model Does

This project implements a deep learning model that performs ECG beat classification using 1D Convolutional Neural Networks (CNNs). It is trained on the MIT-BIH Arrhythmia Database.
✅ Prediction Task

For each input heartbeat (a segment of 187 ECG signal values), the model predicts its class from the following categories:
Label	Description
0	Normal beat
1	Supraventricular ectopic beat (S)
2	Ventricular ectopic beat (V)
3	Fusion beat
4	Unknown or unclassifiable beat
🚨 Detection Capability

By predicting the beat class, the model can detect:

  Irregular heart rhythms (arrhythmias)

  Ectopic beats (abnormal origin beats)

  Potential signs of cardiac distress or failure

This can be used to power real-time ECG monitoring systems that trigger alerts when abnormal heart activity is detected.
💡 Use Cases

  Wearable health monitors

  Hospital ECG alert systems

  Real-time heart rhythm classification

  Personal health analytics apps
