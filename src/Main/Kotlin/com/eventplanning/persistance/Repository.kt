package com.eventplanning.persistance

import com.eventplanning.domain.Event
import com.eventplanning.domain.Participant
import com.eventplanning.domain.Venue


interface Repository {

    /**
     * Establishes a connection to the data store.
     * load the drivers and open/create db file.
     */
    fun connect()

    /**
     * Closes the connection to the data store.
     */
    fun disconnect()

    /**
     *prepares the data store by creating necessary tables if they do not exist.
     * make sure venues, participants, and events tables are set up.
     */
    fun initializeStorage()

    /**
     * Saves a venue to the data store.
     */
    fun saveVenue(venue: Venue): Boolean

    /**
     * Deletes a venue from the data store by its ID.
     */
    fun deleteVenue(id: String): Boolean

    /**
     * Loads all venues from the data store.
     */
    fun loadAllVenues(): List<Venue>


    /**
     * Saves a participant to the data store.
     */
    fun saveParticipant(participant: Participant): Boolean

    /**
     * Deletes a participant from the data store by its ID.
     */
    fun deleteParticipant(id: String): Boolean

    /**
     * Loads all participants from the data store.
     */
    fun loadAllParticipants(): List<Participant>


    /**
     * Saves an event to the data store.
     */
    fun saveEvent(event: Event): Boolean

    /**
     * Deletes an event from the data store by its ID.
     */
    fun deleteEvent(id: String): Boolean

    /**
     * Loads all events from the data store.
     * Uses lookup functions to resolve venue and participant references.
     */
    fun loadAllEvents(
        venueLookup: (String) -> Venue?,
        participantLookup: (String) -> Participant?
    ): List<Event>
}