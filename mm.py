import tensorflow as tf
import pkg_resources

# Load the model
model_path = 'MRImodel.keras'
model = tf.keras.models.load_model(model_path)

# Print model summary to see its architecture
model.summary()

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# List all installed packages and their versions
installed_packages = pkg_resources.working_set
installed_packages_list = sorted([f"{i.key}=={i.version}" for i in installed_packages])
for pkg in installed_packages_list:
    print(pkg)

# Save the list of installed packages to a file
with open('installed_packages.txt', 'w') as f:
    for pkg in installed_packages_list:
        f.write(f"{pkg}\n")
