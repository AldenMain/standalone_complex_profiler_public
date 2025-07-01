import os
import yaml
import shutil
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

DRAFTS_DIR = './outputs/cluster_labels/'
FINALS_DIR = './outputs/cluster_labels/finals/'
BACKUPS_DIR = './outputs/cluster_backups/'

os.makedirs(FINALS_DIR, exist_ok=True)
os.makedirs(BACKUPS_DIR, exist_ok=True)

summary = []

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if data is None:
        print(f"{Fore.RED}[!] Warning: {path} is empty or invalid. Skipping.")
        return None
    return data

def save_yaml(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def review_label(file_name, default_accept=True):
    path = os.path.join(DRAFTS_DIR, file_name)
    data = load_yaml(path)
    if data is None:
        summary.append((file_name, 'Skipped - Invalid'))
        return

    print(f"\n{Fore.CYAN}=== Reviewing Cluster ===")
    print(f"{Fore.YELLOW}Cluster ID: {data.get('cluster_id')}")
    print(f"Label: {data.get('label')}")
    print(f"Dominant Traits: {data.get('traits')}")
    print(f"Psychological Structure:\n{data.get('structure')}")
    print(f"{Fore.CYAN}=========================")

    if default_accept:
        action = input(f"\n{Fore.GREEN}Action? [Enter] Accept / [E]dit / [S]kip / [Q]uit: ").strip().lower()
        if action == '':
            action = 'a'  # Default to accept
    else:
        action = input(f"\n{Fore.GREEN}Action? [A]ccept / [E]dit / [S]kip / [Q]uit: ").strip().lower()

    if action == 'a':
        final_path = os.path.join(FINALS_DIR, file_name)
        if os.path.exists(final_path):
            print(f"{Fore.RED}[!] Warning: {final_path} already exists.")
            overwrite = input("Overwrite? [y/N]: ").strip().lower()
            if overwrite != 'y':
                print(f"{Fore.YELLOW}Skipped to avoid overwrite.")
                summary.append((file_name, 'Skipped - Exists'))
                return
        shutil.copy(path, os.path.join(BACKUPS_DIR, file_name))
        shutil.move(path, final_path)
        print(f"{Fore.GREEN}Accepted and moved to {FINALS_DIR}")
        summary.append((file_name, 'Accepted'))

    elif action == 'e':
        new_label = input("New Label (leave blank to keep): ").strip()
        new_traits = input("New Traits (comma-separated, leave blank to keep): ").strip()
        new_structure = input("New Structure Narrative (leave blank to keep): ").strip()

        if new_label:
            data['label'] = new_label
        else:
            print(f"{Fore.YELLOW}[!] Label unchanged.")

        if new_traits:
            data['traits'] = [trait.strip() for trait in new_traits.split(',') if trait.strip()]
        else:
            print(f"{Fore.YELLOW}[!] Traits unchanged.")

        if new_structure:
            data['structure'] = new_structure
        else:
            print(f"{Fore.YELLOW}[!] Structure unchanged.")

        save_yaml(path, data)
        print(f"{Fore.GREEN}Edited and saved. Rerun review if needed.")
        summary.append((file_name, 'Edited'))

    elif action == 's':
        print(f"{Fore.YELLOW}Skipped.")
        summary.append((file_name, 'Skipped'))

    elif action == 'q':
        print(f"{Fore.RED}Exiting review session early.")
        raise SystemExit

    else:
        print(f"{Fore.RED}Invalid input. Skipped.")
        summary.append((file_name, 'Skipped - Invalid Input'))

def main():
    draft_files = [f for f in os.listdir(DRAFTS_DIR) if f.endswith('.yaml') and 'draft' in f.lower()]

    if not draft_files:
        print(f"{Fore.RED}No drafts to review.")
        return

    # Ask for batch mode or normal mode
    batch_choice = input(f"\n{Fore.CYAN}Use Batch Accept Mode? [Y/n]: ").strip().lower()
    batch_accept = True if batch_choice in ['', 'y', 'yes'] else False

    for file_name in draft_files:
        review_label(file_name, default_accept=batch_accept)

    print(f"\n{Fore.CYAN}=== Review Summary ===")
    for f, status in summary:
        print(f"{status}: {f}")

    # Optional: Save the summary to a text file
    with open('./outputs/review_summary.txt', 'w', encoding='utf-8') as f_out:
        for file_name, status in summary:
            f_out.write(f"{status}: {file_name}\n")
    print(f"{Fore.GREEN}Summary saved to ./outputs/review_summary.txt")

if __name__ == "__main__":
    main()
