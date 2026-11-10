from .adapter_registry import register_adapter
from .transformer import TransformerAdapter


@register_adapter("attention_baseline")
class PureAttentionAdapter(TransformerAdapter):
    """纯注意力对照适配器。

    只覆盖模型类型标识；加载、分词、prefill、step、状态导出全部复用
    TransformerAdapter，避免复制 Qwen3.5 的混合架构逻辑。
    """

    model_type = "attention_baseline"
