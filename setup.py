from setuptools import setup, find_packages


VERSION = '0.3.0'


install_requires = [
  'torch>=2.0.0',
  'einops>=0.6.1'
]


extras_require = dict()
extras_require["triton"] = [
  "triton>=2.0.0"
]
extras_require["surgery"] = [
  "mosaicml>=0.16.0",
  "transformers>=4.11,<4.33"
]


setup(
  name='flash-attention-softmax-n',
  packages=find_packages(exclude=['tests*']),
  version=VERSION,
  license='GPLv3',
  description='CUDA and Triton implementations of Flash Attention with SoftmaxN.',
  author='Christopher W. Murphy',
  author_email='murphtron5000@gmail.com',
  url='https://github.com/softmax1/Flash-Attention-Softmax-N',
  python_requires=">=3.9",
  long_description=open("README.md", "r", encoding="utf-8").read(),
  long_description_content_type='text/markdown',
  keywords=[
    'artificial intelligence',
    'attention mechanism',
    'transformers'
  ],
  install_requires=install_requires,
  extras_require=extras_require,
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11'
  ],
)
