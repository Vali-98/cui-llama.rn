#!/usr/bin/env node

const fs = require('fs')
const path = require('path')
const { spawnSync } = require('child_process')

const ROOT = path.resolve(__dirname, '../..')
const ARTIFACTS_DIR = path.join(__dirname, 'artifacts')

function run(command, args, options = {}) {
    console.log('')
    console.log(`> ${command} ${args.join(' ')}`)
    console.log('')

    const result = spawnSync(command, args, {
        cwd: ROOT,
        stdio: 'inherit',
        ...options,
    })

    if (result.error) {
        throw result.error
    }

    if (result.status !== 0) {
        throw new Error(
            `Command failed with exit code ${result.status}: ${command}`
        )
    }
}

function main() {
    console.log('========================================')
    console.log(' ChatterUI Native Release Build')
    console.log('========================================')

    // --------------------------------------------------------
    // Clean artifacts
    // --------------------------------------------------------

    console.log('')
    console.log('Cleaning release artifacts...')

    fs.rmSync(ARTIFACTS_DIR, {
        recursive: true,
        force: true,
    })

    fs.mkdirSync(ARTIFACTS_DIR, {
        recursive: true,
    })

    // --------------------------------------------------------
    // Apply ChatterUI patches
    // --------------------------------------------------------

    console.log('')
    console.log('Applying ChatterUI patches...')

    run(
        process.execPath,
        [
            path.join(ROOT, 'chatterui', 'apply-patches.js'),
        ]
    )

    // --------------------------------------------------------
    // Android native libraries
    // --------------------------------------------------------

    console.log('')
    console.log('Building Android native libraries...')

    run(
        'npm',
        ['run', 'build:android-libs']
    )

    const androidArchive = path.join(
        ARTIFACTS_DIR,
        'llama-rn-android-jni-libs.tar.gz'
    )

    console.log('')
    console.log('Packaging Android JNI libraries...')

    run(
        'tar',
        [
            '-czf',
            androidArchive,
            'android/src/main/jniLibs',
        ]
    )

    // --------------------------------------------------------
    // iOS native frameworks
    // --------------------------------------------------------

    console.log('')
    console.log('Building iOS frameworks...')

    run(
        'npm',
        ['run', 'build:ios-frameworks']
    )

    const iosArchive = path.join(
        ARTIFACTS_DIR,
        'llama-rn-ios-xcframework.tar.gz'
    )

    console.log('')
    console.log('Packaging iOS XCFramework...')

    run(
        'tar',
        [
            '-czf',
            iosArchive,
            'ios/rnllama.xcframework',
        ],
        {
            env: {
                ...process.env,
                COPYFILE_DISABLE: '1',
            },
        }
    )

    // --------------------------------------------------------
    // Native artifact manifest
    // --------------------------------------------------------

    console.log('')
    console.log('Writing native artifact manifest...')

    run(
        process.execPath,
        [
            path.join(
                ROOT,
                'install',
                'write-native-artifacts-manifest.js'
            ),
        ]
    )

    // --------------------------------------------------------
    // Done
    // --------------------------------------------------------

    console.log('')
    console.log('========================================')
    console.log(' Build complete')
    console.log('========================================')
    console.log('')

    console.log('Artifacts:')

    // eslint-disable-next-line no-restricted-syntax
    for (const file of fs.readdirSync(ARTIFACTS_DIR)) {
        const fullPath = path.join(ARTIFACTS_DIR, file)
        const stat = fs.statSync(fullPath)

        if (stat.isFile()) {
            console.log(`  ${path.relative(ROOT, fullPath)}`)
        }
    }

    console.log('')
}

try {
    main()
} catch (error) {
    console.error('')
    console.error('========================================')
    console.error(' BUILD FAILED')
    console.error('========================================')
    console.error('')
    console.error(error.message)
    console.error('')

    process.exitCode = 1
}