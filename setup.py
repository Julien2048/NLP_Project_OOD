from setuptools import setup, find_namespace_packages

if __name__ == "__main__":
    setup(
        name="nlp_ood_project",
        description="Package used for a NLP project about OOD",
        package_dir={"": "src"},
        author="Julien Mereau and Agathe Minaro",
        packages=find_namespace_packages('./src'),
        python_requires=">=3.9",
    )