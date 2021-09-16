import json
from IPython.core.display import display, HTML, Javascript
import os
from .util import format_special_chars, format_attention


def model_view_question(attentions, tokens, options):
    """Render model view

        Args:
            attention: list of ``torch.FloatTensor``(one for each layer) of shape
                ``(batch_size(must be 1), num_heads, sequence_length, sequence_length)``
            tokens: list of tokens
            sentence_b_index: index of first wordpiece in sentence B if input text is sentence pair (optional)
            prettify_tokens: indicates whether to remove special characters in wordpieces, e.g. Ġ
    """

    vis_html = """
        <span style="user-select:none">
            Attention: <select id="filter">
              <option value="a">Question -> Answer A</option>
              <option value="b">Question -> Answer B</option>
              <option value="c">Question -> Answer C</option>
              <option value="d">Question -> Answer D</option>
            </select>
        </span>
        <div id='vis'></div>
        """
    display(HTML(vis_html))
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    vis_js = open(os.path.join(__location__, 'model_view.js')).read()

    attn_a = format_attention(attentions['a'])
    attn_b = format_attention(attentions['b'])
    attn_c = format_attention(attentions['c'])
    attn_d = format_attention(attentions['d'])
    tokens_a = tokens['a']
    tokens_b = tokens['b']
    tokens_c = tokens['c']
    tokens_d = tokens['d']
    max_length = max(tokens, key=lambda x: len(tokens[x]))
    start_a = options['a']
    start_b = options['b']
    start_c = options['c']
    start_d = options['d']
    
    slice_a = slice(0, start_a)  # Positions corresponding to question in option a
    slice_a_option = slice(start_a, len(tokens_a))  # Position corresponding to anwer in option a
    slice_b = slice(0, start_b)  # Positions corresponding to question in option b
    slice_b_option = slice(start_b, len(tokens_b))  # Position corresponding to anwer in option b
    slice_c = slice(0, start_c)  # Positions corresponding to question in option c
    slice_c_option = slice(start_c, len(tokens_c))  # Position corresponding to anwer in option c
    slice_d = slice(0, start_d)  # Positions corresponding to question in option d
    slice_d_option = slice(start_d, len(tokens_d))  # Position corresponding to anwer in option d
    
    attn_data = {}
    attn_data['a'] = {
        'attn': attn_a[:, :, slice_a, slice_a_option].tolist(),
        'left_text': tokens_a[slice_a],
        'right_text': tokens_a[slice_a_option]
    }
    attn_data['b'] = {
        'attn': attn_b[:, :, slice_b, slice_b_option].tolist(),
        'left_text': tokens_b[slice_b],
        'right_text': tokens_b[slice_b_option]
    }
    attn_data['c'] = {
        'attn': attn_c[:, :, slice_c, slice_c_option].tolist(),
        'left_text': tokens_c[slice_c],
        'right_text': tokens_c[slice_c_option]
    }
    attn_data['d'] = {
        'attn': attn_d[:, :, slice_d, slice_d_option].tolist(),
        'left_text': tokens_d[slice_d],
        'right_text': tokens_d[slice_d_option]
    }
    
    params = {
        'attention': attn_data,
        'default_filter': "a",
        'max_length': max_length
    }
    
    display(Javascript('window.params = %s' % json.dumps(params)))
    display(Javascript(vis_js))