#!/usr/bin/env node

const fs = require('fs')
const path = require('path')
const { spawnSync } = require('child_process')

const ROOT = path.resolve(__dirname, '../..')
const ARTIFACTS_DIR = path.join(__dirname, 'artifacts')

const ANDROID_ARCHIVE = 'llama-rn-android-jni-libs.tar.gz'
const IOS_ARCHIVE = 'llama-rn-ios-xcframework.tar.gz'

const ANDROID_ARCHIVE_PATH = path.join(ROOT, ANDROID_ARCHIVE)
const IOS_ARCHIVE_PATH = path.join(ROOT, IOS_ARCHIVE)

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

function removeFile(filePath) {
    fs.rmSync(filePath, {
        force: true,
    })
}

function main() {
    console.log('========================================')
    console.log(' ChatterUI Native Release Build')
    console.log('========================================')

    // --------------------------------------------------------
    // Clean release artifacts
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

    // Remove temporary root-level archives from a previous build.
    removeFile(ANDROID_ARCHIVE_PATH)
    removeFile(IOS_ARCHIVE_PATH)

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

    console.log('')
    console.log('Packaging Android JNI libraries...')

    run(
        'tar',
        [
            '-czf',
            ANDROID_ARCHIVE,
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

    console.log('')
    console.log('Packaging iOS XCFramework...')

    run(
        'tar',
        [
            '-czf',
            IOS_ARCHIVE,
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
    // Generate native artifact manifest
    // --------------------------------------------------------
    //
    // write-native-artifacts-manifest.js expects the archives
    // to exist at the package root, so deliberately leave them
    // there until this step has completed.
    // --------------------------------------------------------

    console.log('')
    console.log('Writing native artifact manifest...')

    run(
        process.execPath,
        [
            path.join(
                ROOT,
                'install',
                'write-native-artifacts-manifest.js',
            ),
        ]
    )

    // --------------------------------------------------------
    // Move artifacts into chatterui/release/artifacts
    // --------------------------------------------------------

    console.log('')
    console.log('Moving release artifacts...')

    fs.renameSync(
        ANDROID_ARCHIVE_PATH,
        path.join(ARTIFACTS_DIR, ANDROID_ARCHIVE),
    )

    fs.renameSync(
        IOS_ARCHIVE_PATH,
        path.join(ARTIFACTS_DIR, IOS_ARCHIVE),
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
    console.error('Release artifacts may be incomplete.')
    console.error('')

    process.exitCode = 1
}