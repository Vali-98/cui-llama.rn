import { NativeEventEmitter, DeviceEventEmitter, Platform } from 'react-native'
import type { DeviceEventEmitterStatic } from 'react-native'
import RNLlama from './NativeRNLlama'
import type {
  NativeContextParams,
  NativeLlamaContext,
  NativeCompletionParams,
  NativeCompletionTokenProb,
  NativeCompletionResult,
  NativeTokenizeResult,
  NativeEmbeddingResult,
  NativeSessionLoadResult,
  NativeCPUFeatures,
  NativeEmbeddingParams,
  NativeCompletionTokenProbItem,
  NativeCompletionResultTimings,
} from './NativeRNLlama'
import type {
  SchemaGrammarConverterPropOrder,
  SchemaGrammarConverterBuiltinRule,
} from './grammar'
import { SchemaGrammarConverter, convertJsonSchemaToGrammar } from './grammar'
import type { RNLlamaMessagePart, RNLlamaOAICompatibleMessage } from './chat'
import { formatChat } from './chat'

export type {
  NativeContextParams,
  NativeLlamaContext,
  NativeCompletionParams,
  NativeCompletionTokenProb,
  NativeCompletionResult,
  NativeTokenizeResult,
  NativeEmbeddingResult,
  NativeSessionLoadResult,
  NativeEmbeddingParams,
  NativeCompletionTokenProbItem,
  NativeCompletionResultTimings,
  RNLlamaMessagePart,
  RNLlamaOAICompatibleMessage,
  SchemaGrammarConverterPropOrder,
  SchemaGrammarConverterBuiltinRule,
}

export { SchemaGrammarConverter, convertJsonSchemaToGrammar }

const EVENT_ON_INIT_CONTEXT_PROGRESS = '@RNLlama_onInitContextProgress'
const EVENT_ON_TOKEN = '@RNLlama_onToken'

let EventEmitter: NativeEventEmitter | DeviceEventEmitterStatic
if (Platform.OS === 'ios') {
  // @ts-ignore
  EventEmitter = new NativeEventEmitter(RNLlama)
}
if (Platform.OS === 'android') {
  EventEmitter = DeviceEventEmitter
}

export type TokenData = {
  token: string
  completion_probabilities?: Array<NativeCompletionTokenProb>
}

type TokenNativeEvent = {
  contextId: number
  tokenResult: TokenData
}

export enum GGML_TYPE {
  LM_GGML_TYPE_F32     = 0,
  LM_GGML_TYPE_F16     = 1,
  LM_GGML_TYPE_Q4_0    = 2,
  LM_GGML_TYPE_Q4_1    = 3,
  // LM_GGML_TYPE_Q4_2 = 4, support has been removed
  // LM_GGML_TYPE_Q4_3 = 5, support has been removed
  LM_GGML_TYPE_Q5_0    = 6,
  LM_GGML_TYPE_Q5_1    = 7,
  LM_GGML_TYPE_Q8_0    = 8,
  LM_GGML_TYPE_Q8_1    = 9,
  LM_GGML_TYPE_Q2_K    = 10,
  LM_GGML_TYPE_Q3_K    = 11,
  LM_GGML_TYPE_Q4_K    = 12,
  LM_GGML_TYPE_Q5_K    = 13,
  LM_GGML_TYPE_Q6_K    = 14,
  LM_GGML_TYPE_Q8_K    = 15,
  LM_GGML_TYPE_IQ2_XXS = 16,
  LM_GGML_TYPE_IQ2_XS  = 17,
  LM_GGML_TYPE_IQ3_XXS = 18,
  LM_GGML_TYPE_IQ1_S   = 19,
  LM_GGML_TYPE_IQ4_NL  = 20,
  LM_GGML_TYPE_IQ3_S   = 21,
  LM_GGML_TYPE_IQ2_S   = 22,
  LM_GGML_TYPE_IQ4_XS  = 23,
  LM_GGML_TYPE_I8      = 24,
  LM_GGML_TYPE_I16     = 25,
  LM_GGML_TYPE_I32     = 26,
  LM_GGML_TYPE_I64     = 27,
  LM_GGML_TYPE_F64     = 28,
  LM_GGML_TYPE_IQ1_M   = 29,
  LM_GGML_TYPE_BF16    = 30,
  // LM_GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
  // LM_GGML_TYPE_Q4_0_4_8 = 32,
  // LM_GGML_TYPE_Q4_0_8_8 = 33,
  LM_GGML_TYPE_TQ1_0   = 34,
  LM_GGML_TYPE_TQ2_0   = 35,
  // LM_GGML_TYPE_IQ4_NL_4_4 = 36,
  // LM_GGML_TYPE_IQ4_NL_4_8 = 37,
  // LM_GGML_TYPE_IQ4_NL_8_8 = 38,
  LM_GGML_TYPE_COUNT   = 39,
};


export type ContextParams = Omit<
  NativeContextParams,
  'cache_type_k' | 'cache_type_v' | 'pooling_type'
> & {
  cache_type_k?: GGML_TYPE
  cache_type_v?: GGML_TYPE
  pooling_type?: 'none' | 'mean' | 'cls' | 'last' | 'rank'
}

export type EmbeddingParams = NativeEmbeddingParams

export type CompletionParams = Omit<
  NativeCompletionParams,
  'emit_partial_completion' | 'prompt'
> & {
  prompt?: string
  messages?: RNLlamaOAICompatibleMessage[]
  chatTemplate?: string
}

export type BenchResult = {
  modelDesc: string
  modelSize: number
  modelNParams: number
  ppAvg: number
  ppStd: number
  tgAvg: number
  tgStd: number
}

export class LlamaContext {
  id: number

  gpu: boolean = false

  reasonNoGPU: string = ''

  model: {
    isChatTemplateSupported?: boolean
  } = {}

  constructor({ contextId, gpu, reasonNoGPU, model }: NativeLlamaContext) {
    this.id = contextId
    this.gpu = gpu
    this.reasonNoGPU = reasonNoGPU
    this.model = model
  }

  /**
   * Load cached prompt & completion state from a file.
   */
  async loadSession(filepath: string): Promise<NativeSessionLoadResult> {
    let path = filepath
    if (path.startsWith('file://')) path = path.slice(7)
    return RNLlama.loadSession(this.id, path)
  }

  /**
   * Save current cached prompt & completion state to a file.
   */
  async saveSession(
    filepath: string,
    options?: { tokenSize: number },
  ): Promise<number> {
    return RNLlama.saveSession(this.id, filepath, options?.tokenSize || -1)
  }

  async getFormattedChat(
    messages: RNLlamaOAICompatibleMessage[],
    template?: string,
  ): Promise<string> {
    const chat = formatChat(messages)
    let tmpl = this.model?.isChatTemplateSupported ? undefined : 'chatml'
    if (template) tmpl = template // Force replace if provided
    return RNLlama.getFormattedChat(this.id, chat, tmpl)
  }

  async completion(
    params: CompletionParams,
    callback?: (data: TokenData) => void,
  ): Promise<NativeCompletionResult> {
    let finalPrompt = params.prompt
    if (params.messages) {
      // messages always win
      finalPrompt = await this.getFormattedChat(
        params.messages,
        params.chatTemplate,
      )
    }

    let tokenListener: any =
      callback &&
      EventEmitter.addListener(EVENT_ON_TOKEN, (evt: TokenNativeEvent) => {
        const { contextId, tokenResult } = evt
        if (contextId !== this.id) return
        callback(tokenResult)
      })

    if (!finalPrompt) throw new Error('Prompt is required')
    const promise = RNLlama.completion(this.id, {
      ...params,
      prompt: finalPrompt,
      emit_partial_completion: !!callback,
    })
    return promise
      .then((completionResult) => {
        tokenListener?.remove()
        tokenListener = null
        return completionResult
      })
      .catch((err: any) => {
        tokenListener?.remove()
        tokenListener = null
        throw err
      })
  }

  stopCompletion(): Promise<void> {
    return RNLlama.stopCompletion(this.id)
  }

  tokenizeAsync(text: string): Promise<NativeTokenizeResult> {
    return RNLlama.tokenizeAsync(this.id, text)
  }

  tokenizeSync(text: string): NativeTokenizeResult {
    return RNLlama.tokenizeSync(this.id, text)
  }

  detokenize(tokens: number[]): Promise<string> {
    return RNLlama.detokenize(this.id, tokens)
  }

  embedding(
    text: string,
    params?: EmbeddingParams,
  ): Promise<NativeEmbeddingResult> {
    return RNLlama.embedding(this.id, text, params || {})
  }

  async bench(
    pp: number,
    tg: number,
    pl: number,
    nr: number,
  ): Promise<BenchResult> {
    const result = await RNLlama.bench(this.id, pp, tg, pl, nr)
    const [modelDesc, modelSize, modelNParams, ppAvg, ppStd, tgAvg, tgStd] =
      JSON.parse(result)
    return {
      modelDesc,
      modelSize,
      modelNParams,
      ppAvg,
      ppStd,
      tgAvg,
      tgStd,
    }
  }

  async applyLoraAdapters(
    loraList: Array<{ path: string; scaled?: number }>
  ): Promise<void> {
    let loraAdapters: Array<{ path: string; scaled?: number }> = []
    if (loraList)
      loraAdapters = loraList.map((l) => ({
        path: l.path.replace(/file:\/\//, ''),
        scaled: l.scaled,
      }))
    return RNLlama.applyLoraAdapters(this.id, loraAdapters)
  }

  async removeLoraAdapters(): Promise<void> {
    return RNLlama.removeLoraAdapters(this.id)
  }

  async getLoadedLoraAdapters(): Promise<
    Array<{ path: string; scaled?: number }>
  > {
    return RNLlama.getLoadedLoraAdapters(this.id)
  }

  async release(): Promise<void> {
    return RNLlama.releaseContext(this.id)
  }
}

export async function getCpuFeatures() : Promise<NativeCPUFeatures> {
  return RNLlama.getCpuFeatures()
}

export async function setContextLimit(limit: number): Promise<void> {
  return RNLlama.setContextLimit(limit)
}

let contextIdCounter = 0
const contextIdRandom = () =>
  process.env.NODE_ENV === 'test' ? 0 : Math.floor(Math.random() * 100000)

const modelInfoSkip = [
  // Large fields
  'tokenizer.ggml.tokens',
  'tokenizer.ggml.token_type',
  'tokenizer.ggml.merges',
]
export async function loadLlamaModelInfo(model: string): Promise<Object> {
  let path = model
  if (path.startsWith('file://')) path = path.slice(7)
  return RNLlama.modelInfo(path, modelInfoSkip)
}

const poolTypeMap = {
  // -1 is unspecified as undefined
  none: 0,
  mean: 1,
  cls: 2,
  last: 3,
  rank: 4,
}

export async function initLlama(
  {
    model,
    is_model_asset: isModelAsset,
    pooling_type: poolingType,
    lora,
    lora_list: loraList,
    ...rest
  }: ContextParams,
  onProgress?: (progress: number) => void,
): Promise<LlamaContext> {
  let path = model
  if (path.startsWith('file://')) path = path.slice(7)

  let loraPath = lora
  if (loraPath?.startsWith('file://')) loraPath = loraPath.slice(7)

  let loraAdapters: Array<{ path: string; scaled?: number }> = []
  if (loraList)
    loraAdapters = loraList.map((l) => ({
      path: l.path.replace(/file:\/\//, ''),
      scaled: l.scaled,
    }))

  const contextId = contextIdCounter + contextIdRandom()
  contextIdCounter += 1

  let removeProgressListener: any = null
  if (onProgress) {
    removeProgressListener = EventEmitter.addListener(
      EVENT_ON_INIT_CONTEXT_PROGRESS,
      (evt: { contextId: number; progress: number }) => {
        if (evt.contextId !== contextId) return
        onProgress(evt.progress)
      },
    )
  }

  const poolType = poolTypeMap[poolingType as keyof typeof poolTypeMap]
  const {
    gpu,
    reasonNoGPU,
    model: modelDetails,
  } = await RNLlama.initContext(contextId, {
    model: path,
    is_model_asset: !!isModelAsset,
    use_progress_callback: !!onProgress,
    pooling_type: poolType,
    lora: loraPath,
    lora_list: loraAdapters,
    ...rest,
  }).catch((err: any) => {
    removeProgressListener?.remove()
    throw err
  })
  removeProgressListener?.remove()
  return new LlamaContext({ contextId, gpu, reasonNoGPU, model: modelDetails })
}

export async function releaseAllLlama(): Promise<void> {
  return RNLlama.releaseAllContexts()
}
