#   Copyright [yyyy] [name of copyright owner]
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


setup(
    name="entropica_qaoa",
    version="0.1",
    description="Entropica Labs QAOA package",
    author="Entropica Labs: Jan Lukas Bosse, Ewan Munro",
    packages=find_packages(),
    install_requires=['numpy >= 1.7', 'scipy >= 0.9',
                      'scikit-learn >= 0.16', 'numexpr >= 2.5']
)