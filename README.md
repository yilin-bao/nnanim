# PyTorch Neural Network Visualizer

[![Build Status](https://travis-ci.org/user/repo.svg?branch=master)](https://travis-ci.org/user/repo)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub version](https://badge.fury.io/gh/user/repo.svg)](https://badge.fury.io/gh/user/repo)

![Alt text](Attention.drawio.png)

## Overview

This library is designed to generate visualizations of neural networks by analyzing PyTorch code. Unlike conventional methods, our approach eliminates the need for large parameter files (pth) or specific input formats. With minimal AI knowledge, users can leverage this tool to create visualizations that aid in understanding complex networks, including support for transformers and attention mechanisms.

## Installation

Ensure you have Python and PyTorch installed. Install the library using the following command:

<!-- ```bash
pip install your-library
``` -->

## Usage

To generate visualizations, simply provide PyTorch code as input. The library will analyze the code and create visual representations of the neural network. No need for large parameter files or intricate input formats.

```python
import numpy as np
import nnanim.analyzer
from TestingCode import vit
from TestingCode import modules

la = nnanim.analyzer.ModuleAnalyzer()
la.start_analyze_module(modules.Attention(dim=2*768))
```

For the example `TestingCode`, we are using a code base of vit: https://github.com/gupta-abhay/pytorch-vit/tree/main.

```
['self', 'embedding_layer', 'embedding_layer.projection', 'embedding_layer.projection.0', 'embedding_layer.projection.1', 'embedding_layer.projection.2', 'embedding_layer.projection.3', 'transformer', 'transformer.layers', 'transformer.layers.0', 'transformer.layers.0.0', 'transformer.layers.0.0.norm', 'transformer.layers.0.0.fn', 'transformer.layers.0.0.fn.qkv', 'transformer.layers.0.0.fn.attn_drop', 'transformer.layers.0.0.fn.proj', 'transformer.layers.0.0.fn.proj_drop', 'transformer.layers.0.1', 'transformer.layers.0.1.norm', 'transformer.layers.0.1.fn', 'transformer.layers.0.1.fn.net', 'transformer.layers.0.1.fn.net.0', 'transformer.layers.0.1.fn.net.1', 'transformer.layers.0.1.fn.net.2', 'transformer.layers.0.1.fn.net.3', 'transformer.layers.1', 'transformer.layers.1.0', 'transformer.layers.1.0.norm', 'transformer.layers.1.0.fn', 'transformer.layers.1.0.fn.qkv', 'transformer.layers.1.0.fn.attn_drop', 'transformer.layers.1.0.fn.proj', 'transformer.layers.1.0.fn.proj_drop', 'transformer.layers.1.1', 'transformer.layers.1.1.norm', 'transformer.layers.1.1.fn', 'transformer.layers.1.1.fn.net', 'transformer.layers.1.1.fn.net.0', 'transformer.layers.1.1.fn.net.1', 'transformer.layers.1.1.fn.net.2', 'transformer.layers.1.1.fn.net.3', 'transformer.layers.2', 'transformer.layers.2.0', 'transformer.layers.2.0.norm', 'transformer.layers.2.0.fn', 'transformer.layers.2.0.fn.qkv', 'transformer.layers.2.0.fn.attn_drop', 'transformer.layers.2.0.fn.proj', 'transformer.layers.2.0.fn.proj_drop', 'transformer.layers.2.1', 'transformer.layers.2.1.norm', 'transformer.layers.2.1.fn', 'transformer.layers.2.1.fn.net', 'transformer.layers.2.1.fn.net.0', 'transformer.layers.2.1.fn.net.1', 'transformer.layers.2.1.fn.net.2', 'transformer.layers.2.1.fn.net.3', 'transformer.layers.3', 'transformer.layers.3.0', 'transformer.layers.3.0.norm', 'transformer.layers.3.0.fn', 'transformer.layers.3.0.fn.qkv', 'transformer.layers.3.0.fn.attn_drop', 'transformer.layers.3.0.fn.proj', 'transformer.layers.3.0.fn.proj_drop', 'transformer.layers.3.1', 'transformer.layers.3.1.norm', 'transformer.layers.3.1.fn', 'transformer.layers.3.1.fn.net', 'transformer.layers.3.1.fn.net.0', 'transformer.layers.3.1.fn.net.1', 'transformer.layers.3.1.fn.net.2', 'transformer.layers.3.1.fn.net.3', 'transformer.layers.4', 'transformer.layers.4.0', 'transformer.layers.4.0.norm', 'transformer.layers.4.0.fn', 'transformer.layers.4.0.fn.qkv', 'transformer.layers.4.0.fn.attn_drop', 'transformer.layers.4.0.fn.proj', 'transformer.layers.4.0.fn.proj_drop', 'transformer.layers.4.1', 'transformer.layers.4.1.norm', 'transformer.layers.4.1.fn', 'transformer.layers.4.1.fn.net', 'transformer.layers.4.1.fn.net.0', 'transformer.layers.4.1.fn.net.1', 'transformer.layers.4.1.fn.net.2', 'transformer.layers.4.1.fn.net.3', 'transformer.layers.5', 'transformer.layers.5.0', 'transformer.layers.5.0.norm', 'transformer.layers.5.0.fn', 'transformer.layers.5.0.fn.qkv', 'transformer.layers.5.0.fn.attn_drop', 'transformer.layers.5.0.fn.proj', 'transformer.layers.5.0.fn.proj_drop', 'transformer.layers.5.1', 'transformer.layers.5.1.norm', 'transformer.layers.5.1.fn', 'transformer.layers.5.1.fn.net', 'transformer.layers.5.1.fn.net.0', 'transformer.layers.5.1.fn.net.1', 'transformer.layers.5.1.fn.net.2', 'transformer.layers.5.1.fn.net.3', 'transformer.layers.6', 'transformer.layers.6.0', 'transformer.layers.6.0.norm', 'transformer.layers.6.0.fn', 'transformer.layers.6.0.fn.qkv', 'transformer.layers.6.0.fn.attn_drop', 'transformer.layers.6.0.fn.proj', 'transformer.layers.6.0.fn.proj_drop', 'transformer.layers.6.1', 'transformer.layers.6.1.norm', 'transformer.layers.6.1.fn', 'transformer.layers.6.1.fn.net', 'transformer.layers.6.1.fn.net.0', 'transformer.layers.6.1.fn.net.1', 'transformer.layers.6.1.fn.net.2', 'transformer.layers.6.1.fn.net.3', 'transformer.layers.7', 'transformer.layers.7.0', 'transformer.layers.7.0.norm', 'transformer.layers.7.0.fn', 'transformer.layers.7.0.fn.qkv', 'transformer.layers.7.0.fn.attn_drop', 'transformer.layers.7.0.fn.proj', 'transformer.layers.7.0.fn.proj_drop', 'transformer.layers.7.1', 'transformer.layers.7.1.norm', 'transformer.layers.7.1.fn', 'transformer.layers.7.1.fn.net', 'transformer.layers.7.1.fn.net.0', 'transformer.layers.7.1.fn.net.1', 'transformer.layers.7.1.fn.net.2', 'transformer.layers.7.1.fn.net.3', 'transformer.layers.8', 'transformer.layers.8.0', 'transformer.layers.8.0.norm', 'transformer.layers.8.0.fn', 'transformer.layers.8.0.fn.qkv', 'transformer.layers.8.0.fn.attn_drop', 'transformer.layers.8.0.fn.proj', 'transformer.layers.8.0.fn.proj_drop', 'transformer.layers.8.1', 'transformer.layers.8.1.norm', 'transformer.layers.8.1.fn', 'transformer.layers.8.1.fn.net', 'transformer.layers.8.1.fn.net.0', 'transformer.layers.8.1.fn.net.1', 'transformer.layers.8.1.fn.net.2', 'transformer.layers.8.1.fn.net.3', 'transformer.layers.9', 'transformer.layers.9.0', 'transformer.layers.9.0.norm', 'transformer.layers.9.0.fn', 'transformer.layers.9.0.fn.qkv', 'transformer.layers.9.0.fn.attn_drop', 'transformer.layers.9.0.fn.proj', 'transformer.layers.9.0.fn.proj_drop', 'transformer.layers.9.1', 'transformer.layers.9.1.norm', 'transformer.layers.9.1.fn', 'transformer.layers.9.1.fn.net', 'transformer.layers.9.1.fn.net.0', 'transformer.layers.9.1.fn.net.1', 'transformer.layers.9.1.fn.net.2', 'transformer.layers.9.1.fn.net.3', 'transformer.layers.10', 'transformer.layers.10.0', 'transformer.layers.10.0.norm', 'transformer.layers.10.0.fn', 'transformer.layers.10.0.fn.qkv', 'transformer.layers.10.0.fn.attn_drop', 'transformer.layers.10.0.fn.proj', 'transformer.layers.10.0.fn.proj_drop', 'transformer.layers.10.1', 'transformer.layers.10.1.norm', 'transformer.layers.10.1.fn', 'transformer.layers.10.1.fn.net', 'transformer.layers.10.1.fn.net.0', 'transformer.layers.10.1.fn.net.1', 'transformer.layers.10.1.fn.net.2', 'transformer.layers.10.1.fn.net.3', 'transformer.layers.11', 'transformer.layers.11.0', 'transformer.layers.11.0.norm', 'transformer.layers.11.0.fn', 'transformer.layers.11.0.fn.qkv', 'transformer.layers.11.0.fn.attn_drop', 'transformer.layers.11.0.fn.proj', 'transformer.layers.11.0.fn.proj_drop', 'transformer.layers.11.1', 'transformer.layers.11.1.norm', 'transformer.layers.11.1.fn', 'transformer.layers.11.1.fn.net', 'transformer.layers.11.1.fn.net.0', 'transformer.layers.11.1.fn.net.1', 'transformer.layers.11.1.fn.net.2', 'transformer.layers.11.1.fn.net.3', 'post_transformer_ln', 'cls_layer', 'cls_layer.net', 'cls_layer.net.0']
```

## Documentation

<!-- For in-depth instructions and additional details, refer to the [official documentation](https://your-documentation-link.com). -->

## Contribution Guidelines

Contribute to the project by reporting issues or submitting code improvements. Follow the guidelines for raising issues, creating branches, and submitting pull requests.

## Version History

View the project's version history on the [GitHub Releases](https://github.com/user/repo/releases) page.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Refer to the LICENSE file for more information.

## Project Status

Active development status or any relevant project status updates.

## Community and Contact

Engage with the community on our [forum](https://community-forum-link.com) or contact us via email at [your-email@example.com].

## Badges

Display badges indicating project status, build, version, and other relevant information.

## Screenshots or Demos

Include screenshots or links to demos showcasing the visualizations generated by the library.

## Related Projects

Explore other projects or tools related to neural network visualization.

## Frequently Asked Questions (FAQ)

Address common questions users might have about the library and its usage.

## Acknowledgments

Express gratitude to contributors or projects that have influenced or assisted in the development of this library.
```

Replace placeholders like `user/repo`, `your-documentation-link.com`, `your-library`, and others with your actual information. Customize this template based on the specific details and requirements of your project.
