# Create the EIR-trajectory Elo pools in elo-picker and ingest their folders.
# Run: cd ~/source/activity/elo-picker && bash -l -c "cd ~/source/activity/elo-picker && rvm use 3.4.5 >/dev/null && bundle exec rails runner /home/tim/source/activity/prx-tg/scratch/create_elo_pools.rb"
# Additive and idempotent-ish: finds an existing project by folder, else creates it; ingest
# dedups per project, so a re-run creates no duplicate image rows.

base = "/mnt/nas-ai-models/training-data/prx-tg/diagnostics/deficit-2026-09-28/elo_pools"

begin
  pending = ActiveRecord::Base.connection.migration_context.pending_migrations.map(&:filename)
  puts "pending migrations: #{pending.empty? ? 'none' : pending.inspect}"
rescue StandardError => e
  puts "pending-migration check failed: #{e.class}: #{e.message}"
end

# Why ONE POOL PER VALIDATION INDEX (do not merge them): within a pool every pair is the SAME
# identity at different checkpoints, so Elo ranks render quality. Merging indices makes ~69% of
# pairs cross-identity, and identity variance then inflates `SNR = spread/(2*avg RD)` -- the
# reliability meter would read better while measuring less about quality.
folders = Dir.glob(File.join(base, "eir-traj-*")).select { |f| Dir.exist?(f) }.sort
if folders.empty?
  puts "no pools found under #{base}"
  exit 1
end

folders.each do |folder|
  slug = File.basename(folder)
  project = Project.find_by(folder: folder)
  project ||= Project.create!(
    name: "EIR render quality by checkpoint (#{slug})",
    folder: folder
  )
  created, skipped = IngestImages.call(folder, project: project)
  puts "#{slug}: project_id=#{project.id} created=#{created} skipped=#{skipped} " \
       "images=#{project.images.count}"
end

puts "--- pools ---"
Project.order(:id).each do |p|
  puts "#{p.id}\t#{p.name}\timages=#{p.images.count}\tfolder=#{p.folder}"
end
