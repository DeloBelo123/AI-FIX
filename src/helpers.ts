import { BaseChatModel, BaseOutputParser, ChatGroq, ChatOllama, ChatOpenAI, ChatPromptTemplate, StringOutputParser, StructuredOutputParser } from "./imports";
import { z } from "zod/v3";

export type LLMKind = "groq" | "localOllama" | "vision" 

export type Prettify<T> = {
  [K in keyof T]: T[K]
} & {};

export function logChunk(chunk: string) {
  const flushed = process.stdout.write(chunk)
  if (!flushed) {
    process.stdout.once('drain', () => {})
  }
}

export function createChain(prompt: ChatPromptTemplate, llm: BaseChatModel, parser: BaseOutputParser | null = null) {
  return parser ? prompt.pipe(llm).pipe(parser) : prompt.pipe(llm)
}

export async function wait(ms:number){
  return new Promise(resolve => setTimeout(resolve, ms))
}

export async function *stream(y:string | Array<any>,wait_in_between:number = 1){
  for (const x of y){
    yield x
    await wait(wait_in_between)
  }
}

export function getLLM(kind: LLMKind = "groq") {
  switch (kind) {
    case "groq":
      if (!process.env.CHATGROQ_API_KEY) {
        throw new Error("CHATGROQ_API_KEY is not set");
      }
      return new ChatGroq({
        apiKey: process.env.CHATGROQ_API_KEY,
        model: "llama-3.3-70b-versatile"
      });

    case "vision":
      if (!process.env.OPENROUTER_API_KEY) {
        throw new Error("OPENROUTER_API_KEY is not set");
      }
      return new ChatOpenAI({
        apiKey: process.env.OPENROUTER_API_KEY,
        configuration: {
            baseURL: "https://openrouter.ai/api/v1",
        },
        model: "openai/gpt-4o-mini"
      });

    case "localOllama":
      return new ChatOllama({
        model: "llama3.2:3b"
      });

    default:
      throw new Error("Unknown LLM kind");
  }
}

export async function structure<T extends z.ZodObject<any, any>>({
    data,
    into,
    llm = getLLM("groq"),
    retries = 2
}:{
    data: any,
    into: T,
    llm?: BaseChatModel,
    retries?: number
}): Promise<z.infer<T>> {
    const inputString = typeof data === "string" ? data : JSON.stringify(data, null, 2)
    const jsonParser = StructuredOutputParser.fromZodSchema(into)
    const prompt = await ChatPromptTemplate.fromMessages([
        ["system", `Du bist ein JSON-Formatierer. 
            REGELN:
            - Gib NUR valides JSON zurück, KEIN anderer Text
            - Keine Markdown Code-Blöcke (\`\`\`json)
            - Halte dich EXAKT an das Schema

            Schema:
            {format_instructions}`],
        ["human", "{input}"]
    ]).partial({ format_instructions: jsonParser.getFormatInstructions() })
    const chain = createChain(prompt, llm, jsonParser)
    let lastError: Error | null = null
    for (let i = 0; i <= retries; i++) {
        try {
            const result = await chain.invoke({ input: inputString })
            return into.parse(result)
        } catch (error) {
            lastError = error as Error
            if (i < retries) {
                console.warn(`structure() Versuch ${i + 1} fehlgeschlagen, retry...`)
            }
        }
    }
    throw new Error(`structure() failed after ${retries + 1} attempts, Error: ${lastError?.message}`)
}

/**
 * fasst eine Chat-Konversation zwischen User und Assistant zusammen
 */
export async function summarize({
    conversation,
    fokuss,
    llm = getLLM("groq"),
    maxWords = 150
}: {
    conversation: string,
    fokuss?: string,
    llm?: BaseChatModel,
    maxWords?: number
}): Promise<string> {
    const focusMessage: Array<["system", string]> = fokuss 
        ? [["system", `Fokussiere dich besonders auf die folgenden Themen:\n${fokuss}`]]
        : []
    
    const prompt = ChatPromptTemplate.fromMessages([
        ["system", `Du fasst eine Chat-Konversation zwischen User und Assistant zusammen.
          WICHTIG:
          - Behalte ALLE wichtigen Fakten: Namen, Präferenzen, Entscheidungen, Vereinbarungen
          - Behalte chronologischen Kontext wo relevant für Verständnis
          - Fasse auf max. ${maxWords} Wörter zusammen
          - Format: Kurze, prägnante Zusammenfassung ohne Bullet-Points
          - Ignoriere Small-Talk, fokussiere auf inhaltliche Punkte`],
        ...focusMessage,
        ["human", "{conversation}"]
    ])
    
    const chain = createChain(prompt, llm, new StringOutputParser())
    const result = await chain.invoke({ conversation })
    return typeof result === "string" ? result : String(result)
}
