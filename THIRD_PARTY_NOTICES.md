# Third-party notices

PySynthCon retains this repository's GPL-3.0 license. Dependencies remain under their
own licenses. Data downloads remain subject to their original terms and attribution.

## Figure fonts

The package distributes the original static TrueType files for Source Sans 3
Regular and Semibold (Adobe, version 3.052) and STIX Two Text Semibold (The
STIX Fonts Project Authors, version 2.13 b171). Both families use the SIL Open
Font License 1.1. The README illustrations embed renamed subsets of these
typefaces and carry their copyright and license notices.

- Source Sans 3. [Upstream release](https://github.com/adobe-fonts/source-sans/releases/tag/3.052R) · [License and copyright](src/pysynth/fonts/SourceSans3-LICENSE.md)
- STIX Two Text. [Upstream release](https://github.com/stipub/stixfonts/releases/tag/v2.13b171) · [License and copyright](src/pysynth/fonts/STIXTwoText-OFL.txt)

## Estimators

The SDID numerical solver is a Python adaptation of the synthdid reference
algorithm by David A. Hirshberg and contributors, commit
70c1ce3eac58e28c30b67435ca377bb48baa9b8a. Its BSD notice follows.

Copyright 2019, Stanford University

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


The native Python SCUL adaptation follows Alex Hollingsworth's scul procedure,
commit 121b588cf8e4eebcea68518b57a7c754040153bb. The original R functions are
separately supplied local dependencies and are not distributed here.
The applicable MIT notice follows.

# MIT License

Copyright (c) 2020 Alex Hollingsworth

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


IndividualSC calls the separately installed GPL-licensed MSCMT package through
an independently written process adapter following the documented design;
the CC BY-NC-SA code in IndividualSyntheticControl is not copied or distributed.
The costofcare project is cited as a methodological reference, not vendored.
PySyncon and SCM are runtime adapters; their estimator source code is not copied
here. The historical predictor specifications and numerical reference targets
in replication.py follow pysyncon's example/parity files. Its MIT notice follows.

MIT License

Copyright (c) 2026 Stiofán Fordham

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

