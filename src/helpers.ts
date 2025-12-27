import { BaseChatModel, BaseOutputParser, ChatGroq, ChatOllama, ChatOpenAI, ChatPromptTemplate, StructuredOutputParser } from "./imports";
import { z } from "zod/v3";

export type LLMKind = "groq" | "localOllama" | "vision" 

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

export function createChain(prompt: ChatPromptTemplate, llm: BaseChatModel, parser: BaseOutputParser | null = null) {
    return parser ? prompt.pipe(llm).pipe(parser) : prompt.pipe(llm)
}

/**
 * @example
 * const result = await structure({
 *     data: agentOutput,
 *     into: z.object({ name: z.string(), age: z.number() })
 * })
 */
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
    const chain = createChain(prompt, llm as any, jsonParser)
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
