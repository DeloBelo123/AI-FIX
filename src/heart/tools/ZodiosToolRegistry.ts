import type { Zodios, ZodiosEndpointDescription } from "zodios"
import { DynamicStructuredTool } from "../../imports"
import { z, type ZodTypeAny } from 'zod/v3'

type ZodiosEndpointWithAlias<R> = ZodiosEndpointDescription<R> & {
    name?: string
}

export class ZodiosToolRegistry<T extends readonly ZodiosEndpointWithAlias<any>[]> {
    private apiSchemas: T
    private tools: DynamicStructuredTool[] 
    private zodiosClient: Zodios<T>
    constructor({apiSchemas, zodiosClient}: {apiSchemas: T, zodiosClient: Zodios<T>}){
        this.apiSchemas = apiSchemas
        this.zodiosClient = zodiosClient
        this.tools = this.turnApiIntoTools()
    }
    
    public getTool(name: string): DynamicStructuredTool | undefined {
        const tools = this.tools.filter((tool) => tool.name?.toLowerCase() === name.toLowerCase())
        if (tools.length > 1){
            throw new Error(`Error! es wurden unter dem gleichen namen ${name} mehrere tools registriert!`)
        }
        const tool = tools[0]
        if(!tool){
            console.error(`Tool ${name} not found`)
            return undefined
        }
        return tool
    }

    public getTools(names:string[]){
        return names.map(name => this.getTool(name))
    }

    get Tools(): DynamicStructuredTool[] {
        return this.tools
    }

    private turnApiIntoTools():DynamicStructuredTool[]{
        return this.apiSchemas.map((endpoint)=>{
            return new DynamicStructuredTool({
                name:endpoint.name || `call api ${endpoint.method} ${endpoint.path}`,
                description: endpoint.description|| `calls the api ${endpoint.method} ${endpoint.path}`,
                schema:this.buildToolSchema(endpoint),
                func: async (input) => {
                    return this.zodiosClient.request({
                      method: endpoint.method,
                      url: endpoint.path,
                      params: input?.params,
                      queries: input?.queries,
                      headers: input?.headers,
                      data: input?.body,
                    } as any)
                },
            })
        })
    }

    private buildToolSchema(endpoint: ZodiosEndpointDescription<any>) {
        const queries: Record<string, ZodTypeAny> = {}
        const headers: Record<string, ZodTypeAny> = {}
        let body: ZodTypeAny | undefined

        for (const param of endpoint.parameters ?? []) {
            if (param.type === "Query") {
            queries[param.name] = param.schema as unknown as ZodTypeAny
            }
            if (param.type === "Header") {
            headers[param.name] = param.schema as unknown as ZodTypeAny
            }
            if (param.type === "Body") {
            body = param.schema as unknown as ZodTypeAny
            }
        }

        const schemaShape: Record<string, ZodTypeAny> = {}
        
        if (Object.keys(queries).length > 0) {
            schemaShape.queries = z.object(queries).optional()
        }
        
        if (Object.keys(headers).length > 0) {
            schemaShape.headers = z.object(headers).optional()
        }
        
        if (body) {
            schemaShape.body = body.optional()
        }

        return z.object(schemaShape)
    }

}