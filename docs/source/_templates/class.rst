{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block all_methods %}

   {% if all_methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: {{ name }}
      :template: method.rst

   {% for item in all_methods %}
   {%- if not item.startswith('_') or item in ['__call__', '__getitem__'] %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree: {{ name }}
      :template: attribute.rst

   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
