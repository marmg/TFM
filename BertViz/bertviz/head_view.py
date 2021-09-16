import json
from IPython.core.display import display, HTML, Javascript
import os
from .util import format_special_chars, format_attention


def head_view_question(attentions, tokens, options, prettify_tokens=True):
    """Render head view

        Args:
            attention: list of ``torch.FloatTensor``(one for each layer) of shape
                ``(batch_size(must be 1), num_heads, sequence_length, sequence_length)``
            tokens: list of tokens
            sentence_b_index: index of first wordpiece in sentence B if input text is sentence pair (optional)
            prettify_tokens: indicates whether to remove special characters in wordpieces, e.g. Ġ
    """

    vis_html = """
        <span style="user-select:none">
            Layer: <select id="layer"></select>
            Attention: <select id="filter">
              <option value="a">Question -> Answer A</option>
              <option value="b">Question -> Answer B</option>
              <option value="c">Question -> Answer C</option>
              <option value="d">Question -> Answer D</option>
              <option value="aq"Answer A -> Question</option>
              <option value="bq">Answer B -> Question</option>
              <option value="cq">Answer C -> Question</option>
              <option value="dq">Answer D -> Question</option>
            </select>
            </span>
        <div id='vis'></div>
        """

    display(HTML(vis_html))
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    vis_js = open(os.path.join(__location__, 'head_view.js')).read()

#     if prettify_tokens:
#         tokens = format_special_chars(tokens)

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
    
    attn_data['aq'] = {
        'attn': attn_a[:, :, slice_a_option, slice_a].tolist(),
        'left_text': tokens_a[slice_a_option],
        'right_text': tokens_a[slice_a]
    }
    attn_data['bq'] = {
        'attn': attn_b[:, :, slice_b_option, slice_b].tolist(),
        'left_text': tokens_b[slice_b_option],
        'right_text': tokens_b[slice_b]
    }
    attn_data['cq'] = {
        'attn': attn_c[:, :, slice_c_option, slice_c].tolist(),
        'left_text': tokens_c[slice_c_option],
        'right_text': tokens_c[slice_c]
    }
    attn_data['dq'] = {
        'attn': attn_d[:, :, slice_d_option, slice_d].tolist(),
        'left_text': tokens_d[slice_d_option],
        'right_text': tokens_d[slice_d]
    }
    
    params = {
        'attention': attn_data,
        'default_filter': "a",
        'max_length': max_length
    }

    display(Javascript('window.params = %s' % json.dumps(params)))
    display(Javascript(vis_js))

    return params


def head_view(attention, tokens, sentence_b_start = None, prettify_tokens=True):
    """Render head view

        Args:
            attention: list of ``torch.FloatTensor``(one for each layer) of shape
                ``(batch_size(must be 1), num_heads, sequence_length, sequence_length)``
            tokens: list of tokens
            sentence_b_index: index of first wordpiece in sentence B if input text is sentence pair (optional)
            prettify_tokens: indicates whether to remove special characters in wordpieces, e.g. Ġ
    """

    if sentence_b_start is not None:
        vis_html = """
        <span style="user-select:none">
            Layer: <select id="layer"></select>
            Attention: <select id="filter">
              <option value="all">All</option>
              <option value="aa">Sentence A -> Sentence A</option>
              <option value="ab">Sentence A -> Sentence B</option>
              <option value="ba">Sentence B -> Sentence A</option>
              <option value="bb">Sentence B -> Sentence B</option>
            </select>
            </span>
        <div id='vis'></div>
        """
    else:
        vis_html = """
              <span style="user-select:none">
                Layer: <select id="layer"></select>
              </span>
              <div id='vis'></div> 
            """

    display(HTML(vis_html))
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    vis_js = open(os.path.join(__location__, 'head_view.js')).read()

    if prettify_tokens:
        tokens = format_special_chars(tokens)

    attn = format_attention(attention)
    attn_data = {
        'all': {
            'attn': attn.tolist(),
            'left_text': tokens,
            'right_text': tokens
        }
    }
    if sentence_b_start is not None:
        slice_a = slice(0, sentence_b_start)  # Positions corresponding to sentence A in input
        slice_b = slice(sentence_b_start, len(tokens))  # Position corresponding to sentence B in input
        attn_data['aa'] = {
            'attn': attn[:, :, slice_a, slice_a].tolist(),
            'left_text': tokens[slice_a],
            'right_text': tokens[slice_a]
        }
        attn_data['bb'] = {
            'attn': attn[:, :, slice_b, slice_b].tolist(),
            'left_text': tokens[slice_b],
            'right_text': tokens[slice_b]
        }
        attn_data['ab'] = {
            'attn': attn[:, :, slice_a, slice_b].tolist(),
            'left_text': tokens[slice_a],
            'right_text': tokens[slice_b]
        }
        attn_data['ba'] = {
            'attn': attn[:, :, slice_b, slice_a].tolist(),
            'left_text': tokens[slice_b],
            'right_text': tokens[slice_a]
        }
    params = {
        'attention': attn_data,
        'default_filter': "all"
    }
    attn_seq_len = len(attn_data['all']['attn'][0][0])
    if attn_seq_len != len(tokens):
        raise ValueError(f"Attention has {attn_seq_len} positions, while number of tokens is {len(tokens)}")

    display(Javascript('window.params = %s' % json.dumps(params)))
    display(Javascript(vis_js))