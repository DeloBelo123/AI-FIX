import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_KEY!

if (!supabaseUrl) throw new Error("No NEXT_PUBLIC_SUPABASE_URL in env")
if (!supabaseServiceKey) throw new Error("No SUPABASE_SERVICE_ROLE_KEY or NEXT_PUBLIC_SUPABASE_KEY in env")

export const supabase = createClient(supabaseUrl, supabaseServiceKey)

// Type für verschachtelte Updates mit Type Safety
type NestedUpdate<T> = {
  [K in keyof T]?: T[K] extends object ? T[K] | { [P in keyof T[K]]?: T[K][P] } : T[K]
}; 

export class SupabaseTable<T extends Record<string,any>> {
    public tableName:string
    public structure?:T 
    constructor(tableName:string){
        this.tableName = tableName
    }
    /**
     * @param rows - die neuen Zeilen die du in die Tabelle einfügen möchtest, als Array von Objekten, wo jedes Objekt eine Zeile ist
     * @returns nichst, fügt eifach die neuen Zeilen in die Tabelle ein
     */
    async insert(rows:Array<Partial<T>>){
        const { data:insertedData, error } = await supabase
            .from(this.tableName)
            .insert(rows)
        if (error) {
            throw new Error(`Error inserting data into ${this.tableName}: ${error.message}`);
        }
        return insertedData;
    }
    /**
     * @param columns - die spalten die du abfragen möchtest, standardmäßig ist es "*", also alle spalten,
     * @param where - die filter die du anwenden möchtest was ausgewählt werden soll, standardmäßig ist es ein leeres Array, also keine Filter
     * @param ordered_by - sortierung nach einer spalte (column: spaltenname, descending: true/false)
     * @param limited_to - begrenzt die anzahl der ergebnisse
     * @returns returned ein array von Objekte, wo jedes Objekt eine Zeile der Tabelle ist, die den optionalen Filtern entspricht, wo die Keys die spaltennamen sind und die Values die Werte der Zeile
     */
    // funcitonal overloading
    async select({columns , where, ordered_by, limited_to}:{
        columns:Array<keyof T | "*">, 
        where?:Array<{column:keyof T | (string & {}),is:string | number | boolean | Date | null | undefined}>,
        ordered_by?:{column:keyof T | (string & {}), descending:boolean},
        limited_to?:number,
        first?:false
    }): Promise<Array<Record<keyof T,any>>>
    async select({columns , where, ordered_by, limited_to}:{
        columns:Array<keyof T | "*">, 
        where?:Array<{column:keyof T | (string & {}),is:string | number | boolean | Date | null | undefined}>,
        ordered_by?:{column:keyof T | (string & {}), descending:boolean},
        limited_to?:number,
        first:true
    }): Promise<Record<keyof T,any> | null>
    async select({columns , where, ordered_by, limited_to, first = false}:{
        columns:Array<keyof T | "*">, 
        where?:Array<{column:keyof T | (string & {}),is:string | number | boolean | Date | null | undefined}>,
        ordered_by?:{column:keyof T | (string & {}), descending:boolean},
        limited_to?:number,
        first?:boolean
    }): Promise<Array<Record<keyof T,any>> | Record<any,any> | null>
    {
        let columnString = columns.join(",")
        let query = supabase.from(this.tableName).select(columnString)
        if (where){
            for ( const {column,is} of where) {
                query = query.eq(column as string,is)
            }
        }
        if (ordered_by){
            query = query.order(ordered_by.column as string, { ascending: !ordered_by.descending })
        }
        if (limited_to){
            query = query.limit(limited_to)
        }
        const { data, error } = await query
        if (error) {
            console.error("Error selecting data:", error);
            throw new Error(`Error selecting data from ${this.tableName}: ${error.message}`);
        }
        if(first){
            if(data && data.length > 0){
                return data[0]
            } else {
                console.warn(`No data found in first mode in table: ${this.tableName}.select(${query})`)
                return null
            }
            
        }
        return data
    }
    /**
     * Flatten-Funktion für verschachtelte Objekte (Dot-Notation)
     */
    private flattenNested(obj: Record<string, any>, prefix = ""): Record<string, any> {
        const res: Record<string, any> = {};
        
        for (const key in obj) {
            const val = obj[key];
            const newKey = prefix ? `${prefix}.${key}` : key;
            
            if (val && typeof val === "object" && !Array.isArray(val)) {
                // Rekursiv verschachtelte Objekte flach machen
                Object.assign(res, this.flattenNested(val, newKey));
            } else {
                res[newKey] = val;
            }
        }

        return res;
    }

    /**
     * @param updated - die spalten die du aktualisieren möchtest, als Objekt wo der key der Spaltenname ist und der value der neue Wert
     * @param where - die Filter die genau sagen welche Zeile sich aktualisieren soll, sonst wird jede Zeile aktualisiert!!!
     * @returns die geupdateten Zeilen, also die Zeilen die du aktualisiert hast
     */
    async update({where,update}:{ where:Array<{column:keyof T | (string & {}),is:string | number | boolean | Date | null | undefined}>, update:NestedUpdate<T> }){
        // 1. Objekt flach machen (Dot-Notation für JSON-Properties)
        const flatUpdate = this.flattenNested(update as Record<string, any>);
        
        let query = supabase.from(this.tableName).update(flatUpdate)
        for ( const {column,is} of where){
            query = query.eq(column as string,is)
        }
        const { data, error } = await query
        if (error) {
            throw new Error(`Error updating data in ${this.tableName}: ${error.message}`);
        }
        return data;
    }
    /**
     * @param where - die Filter die genau sagen welche Zeile gelöscht werden soll, sonst wird jede Zeile gelöscht!!!
     * @returns garnichts, führt einfach nur eine Löschaktion aus
     */
    async delete({where}:{ where:Array<{column:keyof T,is:string | number | boolean | Date | null | undefined}> }){
        let query = supabase.from(this.tableName).delete()
        for ( const {column,is} of where){
            query = query.eq(column as string,is)
        }
        const { data, error } = await query
        if (error) {
            throw new Error(`Error deleting data from ${this.tableName}: ${error.message}`);
        }
        return data;
    }
    /**
     * @param where - die Filter die genau sagen welche Werte in die upsert-Daten eingefügt werden sollen (für .eq() Filter)
     * @param upsert - die Daten die du upserten möchtest, als Objekt wo der key der Spaltenname ist und der value der neue Wert
     * @param onConflict - die Spalte die für den Konflikt-Check verwendet wird (normalerweise Primary Key) - verwendet native Supabase .upsert()
     * @returns die upserteten Zeilen, also die Zeilen die du upsertet hast
     */
    async upsert({where,upsert,onConflict}:{ 
        where:Array<{column:keyof T,is:string | number | boolean | Date | null | undefined}>, 
        upsert:Partial<T>,
        onConflict:keyof T | (string & {})
    }){
        if (!onConflict) {
            throw new Error("upsert requires onConflict parameter");
        }

        // Kombiniere where-Werte mit upsert-Daten für vollständiges Objekt
        const combinedData: Partial<T> = { ...upsert };
        for (const { column, is } of where) {
            combinedData[column as keyof T] = is as T[keyof T];
        }

        // Flatten für verschachtelte Objekte (Dot-Notation)
        const flatData = this.flattenNested(combinedData as Record<string, any>);

        // Native Supabase .upsert() - atomar
        const { data, error } = await supabase
            .from(this.tableName)
            .upsert(flatData, {
                onConflict: onConflict as string,
            })
            .select();

        if (error) {
            throw new Error(`Error upserting data in ${this.tableName}: ${error.message}`);
        }

        return data;
    }
}

export function selectTable({tableName,possibleTables}:{tableName:string,possibleTables:Array<SupabaseTable<Record<string,any>>>}){
    const table = possibleTables.find(table => table.tableName === tableName)
    if (!table) {
        throw new Error(`Table:'${tableName}' not found in possibleTables`)
    }
    return table
}







