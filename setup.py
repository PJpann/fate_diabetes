import setuptools

setuptools.setup(
    name='cal_gv_fea',
    version='1.0.0',
    python_requires='>=3',
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'datetime',
        "scipy",
        "cvxopt",
        "numpy",
        "pandas",
        "joblib",
        "tqdm>=4.43",
        "ishneholterlib",
        "numba",
        "astropy",
        'flirt',
        'xgboost',
        'sklearn',
        'pyswarms',
        'matplotlib',
        'seaborn'
    ]
)
