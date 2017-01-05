# Copyright (C) 2016 Scott Rome. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import jinja2

def render(tpl_path, context):
    path, filename = os.path.split(tpl_path)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(path))
    template = env.get_template(filename)
    return template.render(context)

def render_results(phrase, images):

    context = {
	    'phrase': phrase,
	    'images': images,
    }
    result = render('template.html', context)

    with open('output.html', 'w') as f:
        f.write(result)
