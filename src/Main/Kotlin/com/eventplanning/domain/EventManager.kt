package com.eventplanning.domain

import com.eventplanning.persistance.Repository

class EventManager(private val repository: Repository) {
    private val venues = mutableListOf<Venue>()
    private val events = mutableListOf<Event>()
    private val participants = mutableListOf<Participant>()

    /**
     * InitializeData loads all venues, participants, and events from the repository.
     * It returns true if all data is loaded successfully, false otherwise.
     * in event checks for venue and participant references using lookup functions.
     */
    fun initializeData(): Boolean {
        return try {
            venues.clear()
            venues.addAll(repository.loadAllVenues())

            participants.clear()
            participants.addAll(repository.loadAllParticipants())

            events.clear()
            events.addAll(repository.loadAllEvents(
                venueLookup = { id -> getVenueById(id) },
                participantLookup = { id -> getParticipantById(id) }
            ))
            true
        } catch (e: Exception) {
            System.err.println("Error initializing data: ${e.message}")
            false
        }
    }

    /**
     * SaveAllData saves all the data back to the data store.
     * It works by either using the:
     * save database button in the UI or during shutdown hook.
     * loops each collection and saves each item via the repository.
     * It returns true if all data is saved successfully, false otherwise.
     */
    fun saveAllData(): Boolean {
        var success = true
        venues.forEach { if (!repository.saveVenue(it)) success = false }
        participants.forEach { if (!repository.saveParticipant(it)) success = false }
        events.forEach { if (!repository.saveEvent(it)) success = false }
        return success
    }

    /**
     * Adds a new venue to the system
     * Checks for duplicate venue IDs before adding.
     * Saves the venue to the data store via the repository.
     * Returns true if added successfully, false otherwise.
     */
    fun addVenue(venue: Venue): Boolean {
        if (venues.any { it.id == venue.id }) return false
        if (!repository.saveVenue(venue)) return false
        venues.add(venue)
        return true
    }

    /**
     * Deletes a venue from the system.
     * Checks if any events are scheduled at the venue before deleting.
     * If events exist at the venue, deletion is not allowed.
     * Returns true if deleted successfully, false otherwise.
     */
    fun deleteVenue(venue: Venue): Boolean {
        if (events.any { it.venue.id == venue.id }) return false
        if (repository.deleteVenue(venue.id)) {
            venues.removeIf { it.id == venue.id }
            return true
        }
        return false
    }

    /**
     * Adds a new participant to the system.
     * Checks for duplicate participant IDs before adding.
     * Saves the participant to the data store via the repository.
     * Returns true if added successfully, false otherwise.
     */
    fun addParticipant(participant: Participant): Boolean {
        if (participants.any { it.id == participant.id }) return false
        if (!repository.saveParticipant(participant)) return false
        participants.add(participant)
        return true
    }

    /**
     * Deletes a participant from the system.
     * Unregisters the participant from all events before deletion.
     * Returns true if deleted successfully, false otherwise.
     */
    fun deleteParticipant(participant: Participant): Boolean {
        if (repository.deleteParticipant(participant.id)) {
            participants.removeIf { it.id == participant.id }
            events.forEach { it.unregisterParticipant(participant) }
            return true
        }
        return false
    }

    /**
     * Adds a new event to the system.
     * checks if an event with the same ID already exists.
     * checks for scheduling conflicts with existing events at the same venue.
     * Saves the event to the data store via the repository.
     * Returns true if added successfully, false otherwise.
     */
    fun addEvent(event: Event): Boolean {
        if (events.any { it.id == event.id }) return false

        if (events.any { it.conflictsWith(event) }) return false

        if (!repository.saveEvent(event)) return false
        events.add(event)
        return true
    }

    /**
     *if you want to modify an existing event.
     * checks if the updated event's max participants is less than current capacity.
     * checks for scheduling conflicts with other existing events at the same venue.
     * saves the updated event to the data store via the repository.
     * updates the event in the local collection if successful.
     * returns true if modified successfully, false otherwise.
     */
    fun modifyEvent(updatedEvent: Event): Boolean {
        if (updatedEvent.maxParticipants < updatedEvent.getCurrentCapacity()) {
            return false
        }

        val hasConflict = events
            .filter { it.id != updatedEvent.id }
            .any { it.conflictsWith(updatedEvent) }

        if (hasConflict) return false

        if (!repository.saveEvent(updatedEvent)) return false

        val index = events.indexOfFirst { it.id == updatedEvent.id }
        if (index != -1) {
            events[index] = updatedEvent
        } else {
            events.add(updatedEvent)
        }
        return true
    }

    /**
     * updates existing event details in the data store.
     */
    fun updateEvent(event: Event): Boolean = repository.saveEvent(event)

    /**
     * Deletes an event from the system.
     * Removes the event from the data store via the repository.
     * Returns true if deleted successfully, false otherwise.
     */
    fun deleteEvent(event: Event): Boolean {
        if (repository.deleteEvent(event.id)) {
            events.removeIf { it.id == event.id }
            return true
        }
        return false
    }

    /**
     * Retrieval functions for venues, events, and participants.
     * they give read only access to the internal collections.
     * ((tolist)) so callers cannot modify the internal lists directly.
     */
    fun getAllVenues(): List<Venue> = venues.toList()
    fun getVenueById(id: String): Venue? = venues.find { it.id == id }

    fun getAllEvents(): List<Event> = events.toList()
    fun getEventById(id: String): Event? = events.find { it.id == id }

    fun getAllParticipants(): List<Participant> = participants.toList()
    fun getParticipantById(id: String): Participant? = participants.find { it.id == id }
}