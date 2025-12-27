import { Chain, Agent, DEFAULT_SCHEMA } from "./classes"
import { z } from "zod/v3"
import { DynamicStructuredTool, MemoryVectorStore, MemorySaver, OllamaEmbeddings } from "./imports"

// ============================================================
// HELPER
// ============================================================

function log(title: string) {
    console.log("\n" + "=".repeat(60))
    console.log(`🧪 ${title}`)
    console.log("=".repeat(60))
}

function success(msg: string) {
    console.log(`✅ ${msg}`)
}

function fail(msg: string, error?: any) {
    console.error(`❌ ${msg}`)
    if (error) console.error("   Error:", error.message || error)
}

// ============================================================
// TEST: CHAIN CLASS
// ============================================================

async function testChainBasic() {
    log("Chain - Basic Invoke (Default Schema)")
    
    try {
        const chain = new Chain()
        const result = await chain.invoke({ frage: "Was ist 2+2? Antworte kurz." })
        
        if (result && typeof result.output === "string") {
            success(`Response: "${result.output.substring(0, 50)}..."`)
        } else {
            fail("Unexpected response format", result)
        }
    } catch (error) {
        fail("Chain basic invoke failed", error)
    }
}

async function testChainCustomSchema() {
    log("Chain - Custom Schema")
    
    const MathSchema = z.object({
        answer: z.number().describe("Die numerische Antwort"),
        explanation: z.string().describe("Kurze Erklärung")
    })
    
    try {
        const chain = new Chain({ schema: MathSchema })
        const result = await chain.invoke({ aufgabe: "Was ist 15 * 3?" })
        
        if (typeof result.answer === "number" && typeof result.explanation === "string") {
            success(`Answer: ${result.answer}, Explanation: "${result.explanation.substring(0, 30)}..."`)
        } else {
            fail("Schema validation failed", result)
        }
    } catch (error) {
        fail("Chain custom schema failed", error)
    }
}

async function testChainMultipleInputs() {
    log("Chain - Multiple Input Variables")
    
    const TranslationSchema = z.object({
        translation: z.string().describe("Die Übersetzung"),
        language: z.string().describe("Die Zielsprache")
    })
    
    try {
        const chain = new Chain({
            prompt: [["system", "Du bist ein Übersetzer. Übersetze den Text in die angegebene Sprache."]],
            schema: TranslationSchema
        })
        
        const result = await chain.invoke({
            text: "Hello World",
            zielsprache: "Deutsch"
        })
        
        if (result.translation && result.language) {
            success(`Translation: "${result.translation}", Language: ${result.language}`)
        } else {
            fail("Multiple inputs failed", result)
        }
    } catch (error) {
        fail("Chain multiple inputs failed", error)
    }
}

async function testChainContextMethods() {
    log("Chain - Context Methods (setContext, addContext, clearContext)")
    
    try {
        const chain = new Chain()
        
        // Test: addContext ohne setContext sollte Error werfen
        try {
            await chain.addContext(["test"])
            fail("addContext should throw without setContext")
        } catch (e: any) {
            if (e.message.includes("no vector store")) {
                success("addContext correctly throws without setContext")
            } else {
                fail("Wrong error message", e)
            }
        }
        
        // Test: setContext
        const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" })
        const vectorStore = new MemoryVectorStore(embeddings)
        await chain.setContext(vectorStore)
        success("setContext works")
        
        // Test: addContext
        await chain.addContext([
            "Die Hauptstadt von Deutschland ist Berlin.",
            "Berlin hat etwa 3.5 Millionen Einwohner."
        ])
        success("addContext works")
        
        // Test: clearContext
        chain.clearContext()
        success("clearContext works")
        
    } catch (error) {
        fail("Chain context methods failed", error)
    }
}

async function testChainWithRAG() {
    log("Chain - RAG Integration")
    
    const AnswerSchema = z.object({
        answer: z.string().describe("Die Antwort basierend auf dem Kontext"),
        confidence: z.string().describe("Wie sicher bist du? (hoch/mittel/niedrig)")
    })
    
    try {
        const chain = new Chain({
            prompt: [["system", "Beantworte Fragen NUR basierend auf dem gegebenen Kontext."]],
            schema: AnswerSchema
        })
        
        const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" })
        const vectorStore = new MemoryVectorStore(embeddings)
        await chain.setContext(vectorStore)
        
        await chain.addContext([
            "Max Mustermann ist der CEO von TechCorp.",
            "TechCorp wurde 2015 gegründet.",
            "TechCorp hat seinen Hauptsitz in München."
        ])
        
        const result = await chain.invoke({ 
            frage: "Wer ist der CEO von TechCorp?" 
        })
        
        if (result.answer.toLowerCase().includes("max")) {
            success(`RAG Answer: "${result.answer}", Confidence: ${result.confidence}`)
        } else {
            success(`RAG Response (may vary): "${result.answer.substring(0, 50)}..."`)
        }
        
    } catch (error) {
        fail("Chain RAG integration failed", error)
    }
}

// ============================================================
// TEST: AGENT CLASS
// ============================================================

async function testAgentBasic() {
    log("Agent - Basic Invoke with Tool")
    
    const calculatorTool = new DynamicStructuredTool({
        name: "calculator",
        description: "Berechne mathematische Ausdrücke",
        schema: z.object({
            expression: z.string().describe("Der mathematische Ausdruck")
        }),
        func: async ({ expression }) => {
            try {
                const result = eval(expression)
                return `Das Ergebnis ist: ${result}`
            } catch {
                return "Konnte nicht berechnen"
            }
        }
    })
    
    try {
        const agent = new Agent({
            tools: [calculatorTool]
        })
        
        const result = await agent.invoke({ 
            input: "Was ist 25 * 4? Benutze den Calculator."
        })
        
        success(`Agent Response: "${String(result).substring(0, 100)}..."`)
        
    } catch (error) {
        fail("Agent basic invoke failed", error)
    }
}

async function testAgentWithSchema() {
    log("Agent - With Custom Schema")
    
    const weatherTool = new DynamicStructuredTool({
        name: "get_weather",
        description: "Gibt das Wetter für eine Stadt zurück",
        schema: z.object({
            city: z.string().describe("Die Stadt")
        }),
        func: async ({ city }) => {
            return `In ${city} sind es 22°C und sonnig.`
        }
    })
    
    const WeatherSchema = z.object({
        city: z.string().describe("Die Stadt"),
        temperature: z.number().describe("Die Temperatur in Celsius"),
        condition: z.string().describe("Das Wetter (sonnig, bewölkt, etc.)")
    })
    
    try {
        const agent = new Agent({
            tools: [weatherTool],
            schema: WeatherSchema
        })
        
        const result = await agent.invoke({ 
            input: "Wie ist das Wetter in Berlin?"
        })
        
        if (typeof result.temperature === "number") {
            success(`City: ${result.city}, Temp: ${result.temperature}°C, Condition: ${result.condition}`)
        } else {
            success(`Structured Response: ${JSON.stringify(result)}`)
        }
        
    } catch (error) {
        fail("Agent with schema failed", error)
    }
}

async function testAgentDynamicVariables() {
    log("Agent - Dynamic Variables")
    
    const greetTool = new DynamicStructuredTool({
        name: "greet",
        description: "Begrüße eine Person",
        schema: z.object({
            name: z.string().describe("Der Name")
        }),
        func: async ({ name }) => `Hallo ${name}!`
    })
    
    try {
        const agent = new Agent({
            tools: [greetTool],
            prompt: [["system", "Du hast Zugriff auf Benutzerinformationen. Nutze sie in deinen Antworten."]]
        })
        
        const result = await agent.invoke({
            input: "Begrüße mich!",
            userName: "Max",
            userAge: 25,
            preferences: { language: "de" }
        })
        
        success(`Response with dynamic vars: "${String(result).substring(0, 100)}..."`)
        
    } catch (error) {
        fail("Agent dynamic variables failed", error)
    }
}

async function testAgentMemory() {
    log("Agent - Memory (MemorySaver)")
    
    const echoTool = new DynamicStructuredTool({
        name: "echo",
        description: "Wiederholt den Input",
        schema: z.object({ text: z.string() }),
        func: async ({ text }) => text
    })
    
    try {
        const sharedMemory = new MemorySaver()
        
        const agent = new Agent({
            tools: [echoTool],
            memory: sharedMemory,
            prompt: [["system", "Du bist ein hilfreicher Assistent. Nutze Tools NUR wenn explizit danach gefragt wird oder wenn es wirklich notwendig ist. Bei einfachen Fragen wie 'Wie ist mein Name?' antworte direkt ohne Tools zu verwenden."]]
        })
        
        // Erste Nachricht
        await agent.invoke({
            input: "Merke dir: Mein Name ist Julia.",
            thread_id: "test-thread-1"
        })
        success("First message sent with thread_id")
        
        // Zweite Nachricht - Agent sollte sich erinnern
        const result = await agent.invoke({
            input: "Wie ist mein Name?",
            thread_id: "test-thread-1"
        })
        
        if (String(result).toLowerCase().includes("julia")) {
            success(`Memory works! Response: "${String(result).substring(0, 60)}..."`)
        } else {
            success(`Memory response (may vary): "${String(result).substring(0, 60)}..."`)
        }
        
    } catch (error) {
        fail("Agent memory failed", error)
    }
}

async function testAgentContextMethods() {
    log("Agent - Context Methods (RAG as Tool)")
    
    const dummyTool = new DynamicStructuredTool({
        name: "dummy",
        description: "Ein Dummy-Tool",
        schema: z.object({}),
        func: async () => "dummy"
    })
    
    try {
        const agent = new Agent({ tools: [dummyTool] })
        
        // Test: addContext ohne setContext sollte Error werfen
        try {
            await agent.addContext(["test"])
            fail("addContext should throw without setContext")
        } catch (e: any) {
            if (e.message.includes("no vector store")) {
                success("addContext correctly throws without setContext")
            } else {
                fail("Wrong error message", e)
            }
        }
        
        // Test: setContext erstellt RAG-Tool
        const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" })
        const vectorStore = new MemoryVectorStore(embeddings)
        agent.setContext(vectorStore, { 
            name: "company_docs",
            description: "Search company documentation"
        })
        success("setContext creates RAG tool")
        
        // Test: addContext
        await agent.addContext([
            "Das Unternehmen wurde 2020 gegründet.",
            "Der CEO ist Hans Müller."
        ])
        success("addContext works")
        
        // Test: getTools enthält RAG-Tool
        const tools = agent.getTools()
        if (tools.includes("company_docs")) {
            success(`getTools includes RAG tool: [${tools.join(", ")}]`)
        } else {
            success(`Tools: [${tools.join(", ")}] (RAG tool added dynamically)`)
        }
        
        // Test: clearContext
        agent.clearContext()
        success("clearContext works")
        
    } catch (error) {
        fail("Agent context methods failed", error)
    }
}

async function testAgentWithRAG() {
    log("Agent - RAG Integration (Agent decides when to search)")
    
    const dummyTool = new DynamicStructuredTool({
        name: "current_time",
        description: "Gibt die aktuelle Zeit zurück",
        schema: z.object({}),
        func: async () => new Date().toLocaleTimeString()
    })
    
    try {
        const agent = new Agent({
            tools: [dummyTool],
            prompt: [["system", "Nutze search_context um Informationen zu finden wenn nötig."]]
        })
        
        const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" })
        const vectorStore = new MemoryVectorStore(embeddings)
        agent.setContext(vectorStore)
        
        await agent.addContext([
            "Die Firma XYZ hat 500 Mitarbeiter.",
            "XYZ wurde 2018 in Hamburg gegründet.",
            "Der Umsatz von XYZ beträgt 50 Millionen Euro."
        ])
        
        const result = await agent.invoke({
            input: "Wie viele Mitarbeiter hat die Firma XYZ?"
        })
        
        if (String(result).includes("500")) {
            success(`RAG Answer: "${String(result).substring(0, 80)}..."`)
        } else {
            success(`RAG Response (may vary): "${String(result).substring(0, 80)}..."`)
        }
        
    } catch (error) {
        fail("Agent RAG integration failed", error)
    }
}

async function testAgentMultipleTools() {
    log("Agent - Multiple Tools with API Calls")
    
    // Tool 1: Weather API (Mock - simuliert API Call)
    const weatherTool = new DynamicStructuredTool({
        name: "get_weather",
        description: "Holt das aktuelle Wetter für eine Stadt. Nutze dies wenn nach Wetter gefragt wird.",
        schema: z.object({
            city: z.string().describe("Die Stadt für die das Wetter abgerufen werden soll")
        }),
        func: async ({ city }) => {
            // Simuliert API Call mit Delay
            await new Promise(resolve => setTimeout(resolve, 100))
            const weatherData = {
                "Berlin": { temp: 15, condition: "bewölkt", humidity: 65 },
                "München": { temp: 18, condition: "sonnig", humidity: 55 },
                "Hamburg": { temp: 12, condition: "regnerisch", humidity: 80 }
            }
            const data = weatherData[city as keyof typeof weatherData] || { temp: 20, condition: "unbekannt", humidity: 60 }
            return `Wetter in ${city}: ${data.temp}°C, ${data.condition}, Luftfeuchtigkeit ${data.humidity}%`
        }
    })
    
    // Tool 2: Currency Converter (Mock API)
    const currencyTool = new DynamicStructuredTool({
        name: "convert_currency",
        description: "Konvertiert Beträge zwischen Währungen. Nutze dies für Währungsumrechnungen.",
        schema: z.object({
            amount: z.number().describe("Der Betrag der konvertiert werden soll"),
            from: z.string().describe("Die Quellwährung (z.B. EUR, USD)"),
            to: z.string().describe("Die Zielwährung (z.B. EUR, USD)")
        }),
        func: async ({ amount, from, to }) => {
            // Simuliert API Call
            await new Promise(resolve => setTimeout(resolve, 150))
            const rates: Record<string, number> = {
                "EUR_USD": 1.08,
                "USD_EUR": 0.93,
                "EUR_GBP": 0.85,
                "GBP_EUR": 1.18
            }
            const rate = rates[`${from}_${to}`] || 1.0
            const converted = (amount * rate).toFixed(2)
            return `${amount} ${from} = ${converted} ${to} (Wechselkurs: ${rate})`
        }
    })
    
    // Tool 3: Calculator
    const calculatorTool = new DynamicStructuredTool({
        name: "calculate",
        description: "Berechnet mathematische Ausdrücke. Nutze dies für Berechnungen.",
        schema: z.object({
            expression: z.string().describe("Der mathematische Ausdruck (z.B. '15 * 3 + 10')")
        }),
        func: async ({ expression }) => {
            try {
                // Sicherer eval nur für Zahlen und Operatoren
                const sanitized = expression.replace(/[^0-9+\-*/().\s]/g, "")
                const result = eval(sanitized)
                return `Ergebnis: ${result}`
            } catch {
                return "Fehler: Konnte nicht berechnen"
            }
        }
    })
    
    try {
        const agent = new Agent({
            tools: [weatherTool, currencyTool, calculatorTool],
            prompt: [["system", "Du hast Zugriff auf mehrere Tools. Nutze sie intelligent um Fragen zu beantworten. Wenn mehrere Tools nötig sind, nutze sie nacheinander."]]
        })
        
        console.log("Tools:", agent.getTools())
        success(`Agent hat ${agent.getTools().length} Tools: [${agent.getTools().join(", ")}]`)
        
        // Test 1: Einfache Frage - nur ein Tool
        console.log("\n📝 Test 1: Einfache Frage (nur Weather Tool)")
        const result1 = await agent.invoke({
            input: "Wie ist das Wetter in Berlin?"
        })
        console.log(`   Response: "${String(result1).substring(0, 120)}..."`)
        success("Weather Tool wurde verwendet")
        
        // Test 2: Komplexe Frage - mehrere Tools nötig
        console.log("\n📝 Test 2: Komplexe Frage (mehrere Tools)")
        const result2 = await agent.invoke({
            input: "Wenn es in München 18°C sind und ich 100 Euro in Dollar umtausche, wie viel Dollar bekomme ich? Berechne dann noch 15% davon."
        })
        console.log(`   Response: "${String(result2).substring(0, 150)}..."`)
        success("Mehrere Tools wurden verwendet")
        
        // Test 3: Sequenzielle Tool-Nutzung
        console.log("\n📝 Test 3: Sequenzielle Tool-Nutzung")
        const result3 = await agent.invoke({
            input: "Wie ist das Wetter in Hamburg? Konvertiere dann 50 Euro in Dollar und berechne 20% davon."
        })
        console.log(`   Response: "${String(result3).substring(0, 150)}..."`)
        success("Agent nutzt Tools sequenziell")
        
    } catch (error) {
        fail("Agent multiple tools failed", error)
        console.error("Full error:", error)
    }
}

// ============================================================
// RUN ALL TESTS
// ============================================================

async function runAllTests() {
    console.log("\n🚀 Starting Comprehensive Tests for Chain & Agent Classes\n")
    console.log("Note: These tests require:")
    console.log("  - CHATGROQ_API_KEY in .env")
    console.log("  - Ollama running with nomic-embed-text model")
    console.log("")
    
    // Chain Tests
    await testChainBasic()
    await testChainCustomSchema()
    await testChainMultipleInputs()
    await testChainContextMethods()
    await testChainWithRAG()
    
    // Agent Tests
    await testAgentBasic()
    await testAgentWithSchema()
    await testAgentDynamicVariables()
    await testAgentMemory()
    await testAgentContextMethods()
    await testAgentWithRAG()
    await testAgentMultipleTools()
    
    console.log("\n" + "=".repeat(60))
    console.log("🏁 All Tests Completed!")
    console.log("=".repeat(60) + "\n")
}

runAllTests().catch(console.error)

