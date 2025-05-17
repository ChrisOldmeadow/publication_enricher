from setuptools import setup, find_packages

setup(
    name="publication_enricher",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'aiohttp>=3.9.1',
        'pandas>=2.1.0',
        'tqdm>=4.65.0',
        'python-dotenv>=1.0.0',
        'aiosqlite>=0.19.0',
        'tenacity>=8.2.3',
        'aiofiles>=23.2.1'
    ],
    entry_points={
        'console_scripts': [
            'enrich-csv=enrich_csv:main',
        ],
    },
) 