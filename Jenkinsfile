/*******************************************************************************
 * Copyright (c) 2025, National Research Foundation (SARAO)
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use
 * this file except in compliance with the License. You may obtain a copy
 * of the License at
 *
 *   https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/* This Jenkinsfile is specific to the environment at SARAO. It might or might
 * not work anywhere else. It is used only to run the tests; linting and
 * packaging is done with Github Actions.
 */

 pipeline {
   agent {
     dockerfile {
       label 'cuda'
       dir '.ci'
       registryCredentialsId 'dockerhub'  // Supply credentials to avoid rate limit
       args '--runtime=nvidia'
    }
  }

  options {
    timeout(time: 15, unit: 'MINUTES')
  }

  stages {
    stage('Install dependencies') {
      steps {
        sh 'python3 -m venv ./.venv'
        sh '.venv/bin/pip install -r requirements.txt -r requirements-readthedocs.txt'
      }
    }

    stage('Install package') {
      steps {
        sh '.venv/bin/pip install -v .'
      }
    }

    stage('Run tests') {
      steps {
        sh 'PATH="$PWD/.venv/bin:$PATH" pytest -v -ra --junitxml=results.xml --suppress-tests-failed-exit-code'
        junit 'results.xml'
      }
    }
  }
}
