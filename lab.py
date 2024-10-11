import matplotlib.pyplot as plt

# Plot accuracies
plt.figure(figsize=(12, 6))
names = ["ReLU", "Tanh", "Sigmoid", "ActiSwitch(ReLU)", "ActiSwitch(Tanh)", "ActiSwitch(Sigmoid)"]
accuracies = [92, 92.3, 87, 94, 91, 89]
plt.bar(names, accuracies, width=0.4, color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Test Set Accuracies')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.ylim(70, 100)  # Set y-axis limits
plt.tight_layout()
plt.savefig('model_accuracies.png')
plt.show()