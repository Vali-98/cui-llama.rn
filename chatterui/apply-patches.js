#!/usr/bin/env node

const fs = require('fs')
const path = require('path')
const crypto = require('crypto')

const ROOT = path.resolve(__dirname, '..')
const CHATTERUI_DIR = path.join(ROOT, 'chatterui')
const PATCH_DIR = path.join(CHATTERUI_DIR, 'patches')
const MANIFEST_PATH = path.join(PATCH_DIR, 'manifest.json')


// ============================================================
// Utilities
// ============================================================

function readText(file) {
    return fs.readFileSync(file, 'utf8')
}

function sha256(text) {
    return crypto
        .createHash('sha256')
        .update(text)
        .digest('hex')
        .slice(0, 12)
}

function normalize(text) {
    return text
        .replace(/\r\n/g, '\n')
        .replace(/\r/g, '\n')
        .replace(/[\t ]+$/gm, '')
        .replace(/^\s+|\s+$/g, '')
}

function resolveRepoPath(file) {
    return path.resolve(ROOT, file)
}

function resolvePatchPath(file) {
    return path.resolve(PATCH_DIR, file)
}

function fail(message) {
    throw new Error(message)
}

function escapeRegex(value) {
    return value.replace(/[$()*+.?[\\\]^{|}]/g, '\\$&')
}


// ============================================================
// C++ scanner
// ============================================================

function findMatchingBrace(source, openIndex) {
    let depth = 0
    let state = 'normal'

    for (let i = openIndex; i < source.length; i++) {
        const c = source[i]
        const next = source[i + 1]

        if (state === 'normal') {
            if (c === '/' && next === '/') {
                state = 'line-comment'
                i++
                continue
            }

            if (c === '/' && next === '*') {
                state = 'block-comment'
                i++
                continue
            }

            if (c === '"') {
                state = 'string'
                continue
            }

            if (c === "'") {
                state = 'char'
                continue
            }

            if (c === '{') {
                depth++
                continue
            }

            if (c === '}') {
                depth--

                if (depth === 0) {
                    return i
                }
            }

            continue
        }

        if (state === 'line-comment') {
            if (c === '\n') {
                state = 'normal'
            }

            continue
        }

        if (state === 'block-comment') {
            if (c === '*' && next === '/') {
                state = 'normal'
                i++
            }

            continue
        }

        if (state === 'string') {
            if (c === '\\') {
                i++
                continue
            }

            if (c === '"') {
                state = 'normal'
            }

            continue
        }

        if (state === 'char') {
            if (c === '\\') {
                i++
                continue
            }

            if (c === "'") {
                state = 'normal'
            }
        }
    }

    return -1
}

function findMatchingParen(source, openIndex) {
    let depth = 0
    let state = 'normal'

    for (let i = openIndex; i < source.length; i++) {
        const c = source[i]
        const next = source[i + 1]

        if (state === 'normal') {
            if (c === '/' && next === '/') {
                state = 'line-comment'
                i++
                continue
            }

            if (c === '/' && next === '*') {
                state = 'block-comment'
                i++
                continue
            }

            if (c === '"') {
                state = 'string'
                continue
            }

            if (c === "'") {
                state = 'char'
                continue
            }

            if (c === '(') {
                depth++
                continue
            }

            if (c === ')') {
                depth--

                if (depth === 0) {
                    return i
                }
            }

            continue
        }

        if (state === 'line-comment') {
            if (c === '\n') {
                state = 'normal'
            }

            continue
        }

        if (state === 'block-comment') {
            if (c === '*' && next === '/') {
                state = 'normal'
                i++
            }

            continue
        }

        if (state === 'string') {
            if (c === '\\') {
                i++
                continue
            }

            if (c === '"') {
                state = 'normal'
            }

            continue
        }

        if (state === 'char') {
            if (c === '\\') {
                i++
                continue
            }

            if (c === "'") {
                state = 'normal'
            }
        }
    }

    return -1
}

function findFunctionBodyBrace(source, start) {
    let state = 'normal'
    let parenDepth = 0
    let bracketDepth = 0

    for (let i = start; i < source.length; i++) {
        const c = source[i]
        const next = source[i + 1]

        if (state === 'normal') {
            if (c === '/' && next === '/') {
                state = 'line-comment'
                i++
                continue
            }

            if (c === '/' && next === '*') {
                state = 'block-comment'
                i++
                continue
            }

            if (c === '"') {
                state = 'string'
                continue
            }

            if (c === "'") {
                state = 'char'
                continue
            }

            if (c === '(') {
                parenDepth++
                continue
            }

            if (c === ')') {
                parenDepth--
                continue
            }

            if (c === '[') {
                bracketDepth++
                continue
            }

            if (c === ']') {
                bracketDepth--
                continue
            }

            if (
                c === '{' &&
                parenDepth === 0 &&
                bracketDepth === 0
            ) {
                return i
            }

            if (
                c === ';' &&
                parenDepth === 0 &&
                bracketDepth === 0
            ) {
                return -1
            }

            continue
        }

        if (state === 'line-comment') {
            if (c === '\n') {
                state = 'normal'
            }

            continue
        }

        if (state === 'block-comment') {
            if (c === '*' && next === '/') {
                state = 'normal'
                i++
            }

            continue
        }

        if (state === 'string') {
            if (c === '\\') {
                i++
                continue
            }

            if (c === '"') {
                state = 'normal'
            }

            continue
        }

        if (state === 'char') {
            if (c === '\\') {
                i++
                continue
            }

            if (c === "'") {
                state = 'normal'
            }
        }
    }

    return -1
}

function findFunctionStart(source, nameStart) {
    const lineStart = source.lastIndexOf('\n', nameStart - 1) + 1
    return lineStart
}

function findFunctionCandidates(source, functionName) {
    const candidates = []

    const nameRegex = new RegExp(
        `\\b${escapeRegex(functionName)}\\s*\\(`,
        'g'
    )

    let match

    while ((match = nameRegex.exec(source)) !== null) {
        const nameStart = match.index
        const openParen = source.indexOf('(', nameStart)

        if (openParen === -1) {
            continue
        }

        const closeParen = findMatchingParen(source, openParen)

        if (closeParen === -1) {
            continue
        }

        const openBrace = findFunctionBodyBrace(
            source,
            closeParen + 1
        )

        if (openBrace === -1) {
            continue
        }

        const closeBrace = findMatchingBrace(
            source,
            openBrace
        )

        if (closeBrace === -1) {
            fail(
                `Unmatched brace while parsing function "${functionName}".`
            )
        }

        candidates.push({
            start: findFunctionStart(source, nameStart),
            end: closeBrace + 1,
            nameStart,
            openBrace,
            closeBrace,
        })

        nameRegex.lastIndex = closeBrace + 1
    }

    return candidates
}


// ============================================================
// Text patch
// ============================================================

function prepareTextPatch(entry, source, oldText, newText, sourcePath) {
    const normalizedSource = normalize(source)
    const normalizedOld = normalize(oldText)
    const normalizedNew = normalize(newText)

    // First check whether the patch has already been applied.
    //
    // This is useful when the new text itself is unique.
    if (
        normalizedOld !== normalizedNew &&
        normalizedSource.includes(normalizedNew)
    ) {
        const oldCount = countOccurrences(
            normalizedSource,
            normalizedOld
        )

        if (oldCount === 0) {
            return {
                ...entry,
                sourcePath,
                status: 'already-patched',
                source,
            }
        }
    }

    const count = countOccurrences(
        normalizedSource,
        normalizedOld
    )

    if (count === 0) {
        fail(
            `Text patch "${entry.old}" was not found.\n\n` +
            `File:\n` +
            `  ${path.relative(ROOT, sourcePath)}\n\n` +
            `Expected hash:\n` +
            `  ${sha256(normalizedOld)}\n\n` +
            `Refusing to patch.`
        )
    }

    if (count > 1) {
        fail(
            `Text patch "${entry.old}" is ambiguous.\n\n` +
            `File:\n` +
            `  ${path.relative(ROOT, sourcePath)}\n\n` +
            `Found ${count} occurrences.\n\n` +
            `Refusing to patch.`
        )
    }

    return {
        ...entry,
        sourcePath,
        status: 'apply',
        source,
        oldText,
        newText,
    }
}

function countOccurrences(source, value) {
    if (!value) {
        return 0
    }

    let count = 0
    let position = 0

    while (true) {
        const index = source.indexOf(value, position)

        if (index === -1) {
            break
        }

        count++
        position = index + value.length
    }

    return count
}

function applyTextPatch(patch) {
    if (patch.status !== 'apply') {
        return
    }

    const {
        source,
        oldText,
        newText,
        sourcePath,
    } = patch

    const index = source.indexOf(oldText)

    if (index === -1) {
        fail(
            `Text patch disappeared before application:\n` +
            `  ${path.relative(ROOT, sourcePath)}`
        )
    }

    const result =
        source.slice(0, index) +
        newText +
        source.slice(index + oldText.length)

    fs.writeFileSync(sourcePath, result, 'utf8')
}


// ============================================================
// Function patch
// ============================================================

function prepareFunctionPatch(entry, source, oldFunction, newFunction, sourcePath) {
    const candidates = findFunctionCandidates(
        source,
        entry.function
    )

    if (candidates.length === 0) {
        fail(
            `Could not find function "${entry.function}" in:\n` +
            `  ${path.relative(ROOT, sourcePath)}`
        )
    }

    if (candidates.length > 1) {
        fail(
            `Function "${entry.function}" is ambiguous in:\n` +
            `  ${path.relative(ROOT, sourcePath)}\n\n` +
            `Found ${candidates.length} candidates.`
        )
    }

    const candidate = candidates[0]

    const actualFunction = source.slice(
        candidate.start,
        candidate.end
    )

    const normalizedActual = normalize(actualFunction)
    const normalizedOld = normalize(oldFunction)
    const normalizedNew = normalize(newFunction)

    // Already patched.
    if (normalizedActual === normalizedNew) {
        return {
            ...entry,
            sourcePath,
            candidate,
            status: 'already-patched',
            source,
        }
    }

    // Doesn't match upstream.
    if (normalizedActual !== normalizedOld) {
        fail(
            `Upstream implementation of "${entry.function}" has changed.\n\n` +
            `File:\n` +
            `  ${path.relative(ROOT, sourcePath)}\n\n` +
            `Expected old hash:\n` +
            `  ${sha256(normalizedOld)}\n\n` +
            `Found hash:\n` +
            `  ${sha256(normalizedActual)}\n\n` +
            `New implementation hash:\n` +
            `  ${sha256(normalizedNew)}\n\n` +
            `Refusing to patch.`
        )
    }

    return {
        ...entry,
        sourcePath,
        candidate,
        status: 'apply',
        source,
        newFunction,
    }
}

function applyFunctionPatch(patch) {
    if (patch.status !== 'apply') {
        return
    }

    const {
        source,
        candidate,
        newFunction,
        sourcePath,
    } = patch

    const before = source.slice(0, candidate.start)
    const after = source.slice(candidate.end)

    const result =
        before +
        newFunction +
        after

    fs.writeFileSync(sourcePath, result, 'utf8')
}


// ============================================================
// Generic patch preparation
// ============================================================

function preparePatch(entry) {
    const type = entry.type || 'function'

    if (!entry.file) {
        fail('Patch entry is missing "file".')
    }

    if (!entry.old) {
        fail(`Patch entry for "${entry.file}" is missing "old".`)
    }

    if (!entry.new) {
        fail(`Patch entry for "${entry.file}" is missing "new".`)
    }

    const sourcePath = resolveRepoPath(entry.file)
    const oldPath = resolvePatchPath(entry.old)
    const newPath = resolvePatchPath(entry.new)

    if (!fs.existsSync(sourcePath)) {
        fail(
            `Target file does not exist:\n` +
            `  ${path.relative(ROOT, sourcePath)}`
        )
    }

    if (!fs.existsSync(oldPath)) {
        fail(
            `Old patch file does not exist:\n` +
            `  ${path.relative(ROOT, oldPath)}`
        )
    }

    if (!fs.existsSync(newPath)) {
        fail(
            `New patch file does not exist:\n` +
            `  ${path.relative(ROOT, newPath)}`
        )
    }

    const source = readText(sourcePath)
    const oldText = readText(oldPath)
    const newText = readText(newPath)

    switch (type) {
        case 'function':
            if (!entry.function) {
                fail(
                    `Function patch for "${entry.file}" ` +
                    `is missing "function".`
                )
            }

            return prepareFunctionPatch(
                entry,
                source,
                oldText,
                newText,
                sourcePath
            )

        case 'text':
            return prepareTextPatch(
                entry,
                source,
                oldText,
                newText,
                sourcePath
            )

        default:
            fail(
                `Unknown patch type "${type}". ` +
                `Expected "function" or "text".`
            )
    }
}


// ============================================================
// Generic patch application
// ============================================================

function applyPatch(patch) {
    switch (patch.type || 'function') {
        case 'function':
            applyFunctionPatch(patch)
            break

        case 'text':
            applyTextPatch(patch)
            break

        default:
            fail(`Unknown patch type "${patch.type}".`)
    }
}


// ============================================================
// Main
// ============================================================

function main() {
    if (!fs.existsSync(MANIFEST_PATH)) {
        fail(
            `Manifest not found:\n` +
            `  ${path.relative(ROOT, MANIFEST_PATH)}`
        )
    }

    let manifest

    try {
        manifest = JSON.parse(readText(MANIFEST_PATH))
    } catch (error) {
        fail(
            `Failed to parse manifest:\n` +
            `  ${error.message}`
        )
    }

    if (
        !manifest ||
        !Array.isArray(manifest.patches)
    ) {
        fail(
            `Manifest must contain a "patches" array.`
        )
    }

    console.log('ChatterUI patches')
    console.log('')

    const prepared = []

    try {
        // ========================================================
        // Phase 1: validate everything
        // ========================================================

        for (const entry of manifest.patches) {
            const type = entry.type || 'function'

            let label

            if (type === 'function') {
                label =
                    `${entry.file} :: ${entry.function}`
            } else {
                label =
                    `${entry.file} :: ${entry.old} -> ${entry.new}`
            }

            process.stdout.write(`  ${label} ... `)

            const patch = preparePatch(entry)

            prepared.push(patch)

            if (patch.status === 'already-patched') {
                console.log('already patched')
            } else {
                console.log('ready')
            }
        }

        console.log('')

        // ========================================================
        // Phase 2: apply everything
        // ========================================================

        let applied = 0
        let alreadyPatched = 0

        for (const patch of prepared) {
            if (patch.status === 'apply') {
                applyPatch(patch)
                applied++
            } else {
                alreadyPatched++
            }
        }

        if (applied > 0) {
            console.log(`Applied: ${applied}`)
        }

        if (alreadyPatched > 0) {
            console.log(`Already patched: ${alreadyPatched}`)
        }

        if (applied === 0 && alreadyPatched === 0) {
            console.log('Nothing to do.')
        }
    } catch (error) {
        console.error('')
        console.error('ERROR:')
        console.error(error.message)
        console.error('')
        console.error('No files were modified.')
        process.exitCode = 1
    }
}

main()