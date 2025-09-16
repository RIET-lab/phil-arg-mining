#!/usr/bin/env python3
"""
Script to generate prompt template files from the provided templates.
Creates separate files for system prompts and user prompts with placeholders.
Uses configuration file to determine input and output paths.
Handles chevron-enclosed references to other files and zero-shot prompts.
Does not insert data into prompts.

TODO: Add a function to insert paper data into prompts, for use by other modules.

"""
import rootutils
import os
import re
import json
from pathlib import Path
from moralkg import Config

def create_directory(dir_path):
    """Create directory if it doesn't exist."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def write_file(filepath, content):
    """Write content to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content.strip())
    print(f"Created: {filepath}")

def read_template_file(filepath):
    """Read template file content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Template file not found: {filepath}")
        return None

def read_referenced_file(filepath):
    """Read a referenced file content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read() # .strip() # TODO: Add the .strip back in, and add extra newlines as needed to the CoT template files.
    except FileNotFoundError:
        print(f"Warning: Referenced file not found: {filepath}")
        return f"<File not found: {filepath}>"
    except Exception as e:
        print(f"Warning: Error reading referenced file {filepath}: {e}")
        return f"<Error reading file: {filepath}>"

def substitute_references(content, config, zero_shot_prompts=None):
    """Substitute chevron-enclosed references with actual file content."""
    if not content:
        return content
    
    # Handle zero-shot system prompt reference
    if zero_shot_prompts and '<Zero-shot system prompt>' in content:
        zero_shot_system = next((v for k, v in zero_shot_prompts.items() if k.startswith('system_')), '<Zero-shot system prompt not available>')
        content = content.replace('<Zero-shot system prompt>', zero_shot_system)
    
    # Handle JSON schema reference
    if '<model response JSON schema here>' in content:
        schema_path = Path(config.get("argmining.schema"))
        schema_content = read_referenced_file(schema_path)
        content = content.replace('<model response JSON schema here>', schema_content)
    
    # Handle example paper reference
    if '<SCHFAT-43_cleaned.md>' in content:
        paper_path = Path(config.get("paths.snowball.phase_1.prompts.x_shot_examples.paper"))
        paper_content = read_referenced_file(paper_path)
        content = content.replace('<SCHFAT-43_cleaned.md>', paper_content)
    
    # Handle main argument map reference
    if '<SCHFAT-43_main.json>' in content:
        main_map_path = Path(config.get("paths.snowball.phase_1.prompts.x_shot_examples.main_arg_map"))
        main_map_content = read_referenced_file(main_map_path)
        content = content.replace('<SCHFAT-43_main.json>', main_map_content)
    
    # Handle extra argument map reference
    if '<SCHFAT-43_extra.json>' in content:
        extra_map_path = Path(config.get("paths.snowball.phase_1.prompts.x_shot_examples.extra_arg_map"))
        extra_map_content = read_referenced_file(extra_map_path)
        content = content.replace('<SCHFAT-43_extra.json>', extra_map_content)

    # Handle CoT step format instruction references like '<step 1 formatting instructions>'
    # Insert the corresponding step_N.txt content from the CoT step_format_instructions folder.
    cot_step_pattern = re.compile(r"<step\s*(\d+) formatting instructions>", re.IGNORECASE)
    def _replace_cot_step(match):
        step_num = match.group(1)
        step_dir = Path(config.get("paths.snowball.phase_1.prompts.templates.cot_step_format_instructions"))
        step_file = step_dir / f"step_{step_num}.txt"
        return read_referenced_file(step_file)

    content = cot_step_pattern.sub(_replace_cot_step, content)
    
    return content

def parse_template_file(content):
    """Parse template file content to extract system and user prompts."""
    if not content:
        return None
    
    lines = content.split('\n')
    prompts = {}
    current_section = None
    current_content = []
    in_code_block = False
    section_state = None  # 'system' or 'user'
    
    for line in lines:
        stripped = line.strip()
        
        if stripped == 'System Prompt':
            section_state = 'system'
            continue
        elif stripped == 'User Prompt':
            # Save previous section if exists
            if current_section and current_content:
                prompts[current_section] = '\n'.join(current_content).strip()
            section_state = 'user'
            current_section = None
            current_content = []
            continue
        elif re.match(r'^--- (\d+) ---$', stripped) and section_state:
            # Save previous section if exists
            if current_section and current_content:
                prompts[current_section] = '\n'.join(current_content).strip()

            number = stripped[4:-4]  # Extract the number from the line (e.g., "--- 1 ---" -> "1")

            # Determine section based on state and number
            if section_state == 'system':
                current_section = f'system_{number}'
            elif section_state == 'user':
                current_section = f'user_{number}'
            
            current_content = []
            continue
        elif stripped.startswith('```'):
            if in_code_block:
                # End of code block
                in_code_block = False
                continue
            else:
                # Start of code block
                in_code_block = True
                continue
        
        # Collect content when in code block and have a current section
        if in_code_block and current_section:
            current_content.append(line)
        elif not in_code_block and current_section and stripped:
            # Also collect non-code content that's not empty
            current_content.append(line)
    
    # Save the last section
    if current_section and current_content:
        prompts[current_section] = '\n'.join(current_content).strip()
    
    return prompts

def write_prompts_to_directory(prompts, output_dir, shot_type, variation_key):
    """Write parsed prompts to the specified output directory."""
    create_directory(output_dir)
    
    files_written = []
    
    # Get all system prompts dynamically
    system_prompts = {k: v for k, v in prompts.items() if k.startswith('system_')}
    for key in sorted(system_prompts.keys(), key=lambda x: int(x.split('_')[1])):
        prompt_number = key.split('_')[1]
        if variation_key == 'default':
            filepath = output_dir / f"system_prompt_{prompt_number}.txt"
        else:
            filepath = output_dir / f"system_prompt_{prompt_number}_{variation_key}.txt"
        write_file(filepath, prompts[key])
        files_written.append(filepath)
    
    # Get all user prompts dynamically
    user_prompts = {k: v for k, v in prompts.items() if k.startswith('user_')}
    for key in sorted(user_prompts.keys(), key=lambda x: int(x.split('_')[1])):
        prompt_number = key.split('_')[1]
        if len(user_prompts) == 1:
            # If there's only one user prompt, use the simple filename
            filepath = output_dir / "user_prompt.txt"
        else:
            # If there are multiple user prompts, number them
            filepath = output_dir / f"user_prompt_{prompt_number}.txt"
        write_file(filepath, prompts[key])
        files_written.append(filepath)
    
    return files_written

def generate_prompt_variations(prompts, config, zero_shot_prompts, shot_type):
    """Generate prompt variations by substituting references."""
    variations = {}
    
    # If we have multiple zero-shot system prompts, generate variations for one-shot and few-shot
    if shot_type in ['one-shot', 'few-shot'] and zero_shot_prompts:
        zero_shot_system_prompts = {k: v for k, v in zero_shot_prompts.items() if k.startswith('system_')}
        
        if len(zero_shot_system_prompts) > 1:
            # Generate variations for each zero-shot system prompt
            for zs_key, zs_content in zero_shot_system_prompts.items():
                zs_number = zs_key.split('_')[1]
                variation_key = f"zs{zs_number}"
                
                # Create a temporary zero_shot_prompts dict with just this system prompt
                temp_zs_prompts = {zs_key: zs_content}
                
                variation_prompts = {}
                for prompt_key, prompt_content in prompts.items():
                    substituted_content = substitute_references(prompt_content, config, temp_zs_prompts)
                    variation_prompts[prompt_key] = substituted_content
                
                variations[variation_key] = variation_prompts
        else:
            # Only one zero-shot system prompt, create default variation
            default_prompts = {}
            for prompt_key, prompt_content in prompts.items():
                substituted_content = substitute_references(prompt_content, config, zero_shot_prompts)
                default_prompts[prompt_key] = substituted_content
            variations['default'] = default_prompts
    else:
        # For zero-shot or when no zero-shot prompts available, just substitute other references
        default_prompts = {}
        for prompt_key, prompt_content in prompts.items():
            substituted_content = substitute_references(prompt_content, config, None)
            default_prompts[prompt_key] = substituted_content
        variations['default'] = default_prompts
    
    return variations

def write_prompt_variations(variations, output_dirs, shot_type):
    """Write prompt variations to output directories."""
    total_files_written = []
    
    for variation_key, prompts in variations.items():
        output_dir = output_dirs[shot_type]
        if variation_key == 'default':
            print(f"  Writing {variation_key} variation to: {output_dir}")
        else:
            print(f"  Writing {variation_key} variation to: {output_dir} with suffix: _{variation_key}")
        files_written = write_prompts_to_directory(prompts, output_dir, shot_type, variation_key)
        total_files_written.extend(files_written)

    return total_files_written


def process_cot_templates(config, template_dir):
    """Process CoT templates (all_in_one, system_stepwise, user_stepwise).

    This will parse each CoT template, substitute step format instructions into
    both system and user prompts, generate variations (using zero-shot system
    prompts where appropriate), and write files to the config-specified output
    directories under meta_llama_3.cot.
    """
    cot_templates = {
        'all_in_one': template_dir / 'all_in_one.txt',
        'system_stepwise': template_dir / 'system_stepwise.txt',
        'user_stepwise': template_dir / 'user_stepwise.txt'
    }

    # Output directories per strategy: load strategy-specific paths from config
    output_dirs = {
        'all_in_one': Path(config.get('paths.snowball.phase_1.prompts.meta_llama_3.cot.all_in_one')),
        'system_stepwise': Path(config.get('paths.snowball.phase_1.prompts.meta_llama_3.cot.system_stepwise')),
        'user_stepwise': Path(config.get('paths.snowball.phase_1.prompts.meta_llama_3.cot.user_stepwise'))
    }

    total_files = []

    for key, tpl_path in cot_templates.items():
        print(f"\nParsing CoT template: {key}")
        tpl_content = read_template_file(tpl_path)
        if not tpl_content:
            print(f"  Skipping CoT template {key} - file not found")
            continue

        prompts = parse_template_file(tpl_content)
        if not prompts:
            print(f"  Skipping CoT template {key} - parse returned no prompts")
            continue

        # For CoT templates we always substitute step format instructions
        substituted_prompts = {}
        for pkey, pcontent in prompts.items():
            # Do not use any zero-shot prompts for CoT substitution — always use direct substitutions
            substituted = substitute_references(pcontent, config, None)
            substituted_prompts[pkey] = substituted

        # Variations: generate default (no zero-shot) variation for CoT
        variations = generate_prompt_variations(substituted_prompts, config, None, 'cot')

        # Ensure output dir exists
        out_dir = output_dirs.get(key)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        # Write variations to the appropriate CoT output directory. Use a 'cot' key so write_prompt_variations
        # can index the provided output_dirs mapping.
        files_written = write_prompt_variations(variations, { 'cot': out_dir }, 'cot')
        total_files.extend(files_written)

    return total_files


def process_standard_templates(config):
    """Process standard zero/one/few-shot templates and write generated prompts.

    Returns:
        (total_files_written, zero_shot_prompts)
    """
    # Get standard template directory from config
    template_dir = Path(config.get("paths.snowball.phase_1.prompts.templates.standard"))
    if not template_dir.exists():
        print(f"Template directory not found: {template_dir}")
        return [], {}

    # Template file mapping
    template_files = {
        'zero-shot': template_dir / "zero-shot.txt",
        'one-shot': template_dir / "one-shot.txt",
        'few-shot': template_dir / "few-shot.txt"
    }

    # Output directory mapping from config
    output_dirs = {
        'zero-shot': Path(config.get("paths.snowball.phase_1.prompts.meta_llama_3.standard.zero-shot")),
        'one-shot': Path(config.get("paths.snowball.phase_1.prompts.meta_llama_3.standard.one-shot")),
        'few-shot': Path(config.get("paths.snowball.phase_1.prompts.meta_llama_3.standard.few-shot")),
    }

    # Create output dirs if necessary
    for output_dir in output_dirs.values():
        create_directory(output_dir)

    print("Generating prompt template files from configuration...")
    print("=" * 60)
    
    total_files_written = []
    all_parsed_prompts = {}
    
    # First pass: Parse all template files
    for shot_type, template_file in template_files.items():
        print(f"\nParsing {shot_type} templates:")
        print(f"  Reading from: {template_file}")
        
        # Read and parse template file
        template_content = read_template_file(template_file)
        if not template_content:
            print(f"  Skipping {shot_type} - template file not found")
            continue
            
        prompts = parse_template_file(template_content)
        if not prompts or not any(prompts.values()):
            print(f"  Skipping {shot_type} - could not parse template file")
            continue
        
        all_parsed_prompts[shot_type] = prompts
        
        # Show what was parsed
        system_count = len([k for k in prompts.keys() if k.startswith('system_')])
        user_count = len([k for k in prompts.keys() if k.startswith('user_')])
        print(f"  Found {system_count} system prompt(s) and {user_count} user prompt(s)")
    
    # Second pass: Generate variations with reference substitution
    zero_shot_prompts = all_parsed_prompts.get('zero-shot', {})
    
    for shot_type, prompts in all_parsed_prompts.items():
        print(f"\nProcessing {shot_type} variations:")
        
        # Generate prompt variations
        variations = generate_prompt_variations(prompts, config, zero_shot_prompts, shot_type)
        
        print(f"  Generated {len(variations)} variation(s): {list(variations.keys())}")
        
        # Write variations to files
        files_written = write_prompt_variations(variations, output_dirs, shot_type)
        total_files_written.extend(files_written)

    return total_files_written

def main() -> None:
    """Main function to generate all prompt files from config."""
    
    # Load configuration
    try:
        config = Config.load()
        print("Configuration loaded successfully")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Process standard templates (zero/one/few-shot)
    #total_files_written = process_standard_templates(config)
    
    total_files_written = []
    # Additionally process CoT templates (they live in a separate templates folder)
    try:
        cot_template_dir = Path(config.get("paths.snowball.phase_1.prompts.templates.cot"))
        print("\nProcessing CoT templates...")
        cot_files = process_cot_templates(config, cot_template_dir)
        total_files_written.extend(cot_files)
    except Exception as e:
        print(f"Warning: failed to process CoT templates: {e}")
    
    print("\n" + "=" * 60)
    print(f"Prompt generation completed! Generated {len(total_files_written)} files.")


if __name__ == "__main__":
    main()